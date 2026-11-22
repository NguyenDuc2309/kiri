//! Who may use the engine's socket on a Linux host, without signing in again.
//!
//! The socket is 0660 root:kiri, so members of the `kiri` group may use it, as
//! the docker group may use dockerd's. But a group someone was just added to
//! only applies to sessions that start after: until they log out and back in,
//! the window and every terminal they have open are refused. So each member
//! also gets their own entry in the socket's access list (a POSIX ACL), which
//! applies at once, to every session they have.
//!
//! The daemon writes the list when it makes the socket, and `kiri host-access`
//! (run as root, from the window's onboarding through pkexec) writes it again
//! after adding someone.
//!
//! An entry in the list outlives the membership it was written for, so the
//! daemon doesn't rely on it alone: a `Gate` checks every new connection
//! against the group, and every few seconds checks the people still connected,
//! closing the connections of anyone removed (streams and `exec` sessions
//! too) and writing the list again without them. Removal applies as soon as
//! the system's user database says so; where that is cached (nscd, sssd), it
//! is when the cache catches up.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, OsStr};
use std::io;
use std::net::Shutdown;
use std::os::fd::AsRawFd;
use std::os::raw::{c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How often the people still connected are checked again.
const RECHECK: Duration = Duration::from_secs(2);
/// How long a single `getent` call may run before it is killed.
const GETENT_TIMEOUT: Duration = Duration::from_secs(3);
/// Maximum stdout accepted from a single `getent` call.
const GETENT_MAX_STDOUT: usize = 4 * 1024 * 1024;

/// An account or group record and the buffer its strings live in.
struct Rec<T> {
    rec: T,
    _buf: Vec<c_char>,
}

/// Runs a reentrant NSS lookup (getpwuid_r and the like), growing the buffer
/// until the record fits. `Ok(None)`: there is no such account or group.
///
/// # Safety
/// `T` must be `libc::passwd` or `libc::group`, and `call` the matching `_r` lookup.
unsafe fn nss<T>(mut call: impl FnMut(*mut T, *mut c_char, usize, *mut *mut T) -> c_int) -> io::Result<Option<Rec<T>>> {
    let mut size = 1024;
    loop {
        let mut buf: Vec<c_char> = vec![0; size];
        // SAFETY: passwd and group are plain data; all-zero is a valid value.
        let mut rec: T = unsafe { std::mem::zeroed() };
        let mut out: *mut T = std::ptr::null_mut();
        match call(&mut rec, buf.as_mut_ptr(), buf.len(), &mut out) {
            0 if out.is_null() => return Ok(None),
            // The record's strings point into `buf`, whose heap storage moves with it.
            0 => return Ok(Some(Rec { rec, _buf: buf })),
            // Groups of many thousands of names run to megabytes.
            libc::ERANGE if size < 1 << 28 => size *= 2,
            libc::ENOENT | libc::ESRCH => return Ok(None),
            e => return Err(io::Error::from_raw_os_error(e)),
        }
    }
}

/// Trusted locations for `getent`: never searched through `PATH` because this
/// code runs as root in the daemon and in `kiri host-access`.
const GETENT_PATHS: &[&str] = &["/usr/bin/getent", "/bin/getent", "/run/current-system/sw/bin/getent"];

fn find_getent() -> Option<&'static Path> {
    GETENT_PATHS.iter().map(Path::new).find(|p| p.is_file())
}

/// An account: its name, uid and primary group.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Account {
    name: CString,
    uid: u32,
    gid: u32,
}

impl Account {
    fn from_rec(a: &Rec<libc::passwd>) -> Option<Account> {
        if a.rec.pw_name.is_null() {
            return None;
        }
        // SAFETY: pw_name is a NUL-terminated string in the record's buffer.
        let name = unsafe { CStr::from_ptr(a.rec.pw_name) }.to_owned();
        Some(Account { name, uid: a.rec.pw_uid, gid: a.rec.pw_gid })
    }
}

/// A group: its name, gid and explicit member names.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Group {
    name: CString,
    gid: u32,
    members: Vec<CString>,
}

impl Group {
    fn from_rec(g: &Rec<libc::group>) -> Option<Group> {
        if g.rec.gr_name.is_null() {
            return None;
        }
        // SAFETY: gr_name and gr_mem strings are NUL-terminated in the record's buffer.
        let name = unsafe { CStr::from_ptr(g.rec.gr_name) }.to_owned();
        let mut members = Vec::new();
        let mut m = g.rec.gr_mem;
        // SAFETY: gr_mem is a NULL-terminated array of account names for this group.
        unsafe {
            while !m.is_null() && !(*m).is_null() {
                members.push(CStr::from_ptr(*m).to_owned());
                m = m.add(1);
            }
        }
        Some(Group { name, gid: g.rec.gr_gid, members })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LookupResult<T> {
    /// Entry found.
    Found(T),
    /// Definitely absent (no `getent` binary and local miss, or invalid/mismatched key).
    NotFound,
    /// `getent` returned exit 2: either absent or NSS directory temporarily unavailable.
    UnavailableOrNotFound,
}

fn local_account_by_name(name: &CStr) -> io::Result<Option<Account>> {
    // SAFETY: getpwnam_r with a valid C string, a passwd record and a buffer of the stated size.
    Ok(unsafe { nss(|rec, buf, len, out| libc::getpwnam_r(name.as_ptr(), rec, buf, len, out)) }?.and_then(|a| Account::from_rec(&a)))
}

fn local_group_by_name(name: &CStr) -> io::Result<Option<Group>> {
    // SAFETY: getgrnam_r with a valid C string, a group record and a buffer of the stated size.
    Ok(unsafe { nss(|rec, buf, len, out| libc::getgrnam_r(name.as_ptr(), rec, buf, len, out)) }?.and_then(|g| Group::from_rec(&g)))
}

fn account_by_uid(uid: u32) -> io::Result<Option<Account>> {
    // SAFETY: getpwuid_r with a passwd record and a buffer of the stated size.
    match unsafe { nss(|rec, buf, len, out| libc::getpwuid_r(uid, rec, buf, len, out)) }? {
        Some(a) => Ok(Account::from_rec(&a)),
        None => Ok(getent_account(uid.to_string().as_bytes())?.filter(|a| a.uid == uid)),
    }
}

fn account_by_name(name: &CStr) -> io::Result<Option<Account>> {
    match local_account_by_name(name)? {
        Some(a) => Ok(Some(a)),
        None => Ok(getent_account(name.to_bytes())?.and_then(|a| match_account_name(name.to_bytes(), a))),
    }
}

fn valid_key(key: &[u8]) -> bool {
    !key.is_empty() && key[0] != b'-' && !key.iter().any(|&b| matches!(b, b':' | b'\n' | b'\r' | 0))
}

/// Whether `strtoul(key, &end, 10)` in the C locale would consume all of `key`
/// as a numeric id rather than a name (`"0"`, `"+0"`, `" 0"`, `" -0"`, etc.).
fn is_numeric_key(key: &[u8]) -> bool {
    let Some(start) = key.iter().position(|&b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)) else {
        return false;
    };
    let rest = &key[start..];
    let digits = rest.strip_prefix(b"+").or_else(|| rest.strip_prefix(b"-")).unwrap_or(rest);
    !digits.is_empty() && digits.iter().all(u8::is_ascii_digit)
}

fn match_account_name(query: &[u8], account: Account) -> Option<Account> {
    if is_numeric_key(query) && account.name.as_bytes() != query {
        return None;
    }
    Some(account)
}

fn match_group_name(query: &[u8], group: Group) -> Option<Group> {
    if is_numeric_key(query) && group.name.as_bytes() != query {
        return None;
    }
    Some(group)
}

fn single_line(text: &[u8]) -> Option<&[u8]> {
    let line = text.strip_suffix(b"\n").unwrap_or(text);
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line.is_empty() || line.contains(&b'\n') || line.contains(&b'\r') {
        return None;
    }
    Some(line)
}

fn parse_u32(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(bytes).ok()?.parse().ok()
}

/// Parses a GID token from `getent initgroups`: glibc formats `gid_t` with
/// signed `PRId32`, so GIDs >= 2^31 appear as negative decimal integers.
fn parse_initgroups_gid(bytes: &[u8]) -> Option<u32> {
    if let Some(rest) = bytes.strip_prefix(b"-") {
        if rest.is_empty() || !rest.iter().all(u8::is_ascii_digit) {
            return None;
        }
        let v: i32 = std::str::from_utf8(bytes).ok()?.parse().ok()?;
        if v >= 0 {
            return None;
        }
        return Some(v as u32);
    }
    parse_u32(bytes)
}

enum GetentRaw {
    /// The `getent` binary does not exist.
    NoBinary,
    /// `getent` ran and exited with `(status, stdout)`.
    Ran(std::process::ExitStatus, Vec<u8>),
}

/// Runs `program -- <args...>` with a sanitized environment, bounded duration,
/// and bounded stdout capture. Kills the child process group and reaps it on
/// timeout or oversized output.
fn exec_getent(program: &Path, args: &[&OsStr], timeout: Duration, max_bytes: usize) -> io::Result<GetentRaw> {
    use std::io::Read;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(program);
    cmd.env_clear()
        .env("LC_ALL", "C")
        .arg("--")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(GetentRaw::NoBinary),
        Err(e) => return Err(e),
    };

    let kill_and_reap = |child: &mut std::process::Child| {
        let pid = child.id();
        if pid > 0 && pid <= i32::MAX as u32 {
            // SAFETY: child has not been waited yet and leads its own process group.
            unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) };
        }
        let _ = child.kill();
        let _ = child.wait();
    };

    let mut stdout = child.stdout.take().expect("stdout piped");
    let fd = stdout.as_raw_fd();
    // SAFETY: fd is an open pipe descriptor owned by `stdout`.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        let err = io::Error::last_os_error();
        drop(stdout);
        kill_and_reap(&mut child);
        return Err(err);
    }

    let deadline = Instant::now() + timeout;
    let mut out = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut eof = false;

    while !eof {
        let now = Instant::now();
        if now >= deadline {
            drop(stdout);
            kill_and_reap(&mut child);
            return Err(io::Error::new(io::ErrorKind::TimedOut, format!("getent timed out after {}ms", timeout.as_millis())));
        }
        let rem_ms = deadline.saturating_duration_since(now).as_millis().clamp(1, i32::MAX as u128) as libc::c_int;
        let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        // SAFETY: `pfd` points to one valid `pollfd` struct.
        let ready = unsafe { libc::poll(&mut pfd, 1, rem_ms) };
        if ready < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            drop(stdout);
            kill_and_reap(&mut child);
            return Err(err);
        }
        if ready == 0 {
            drop(stdout);
            kill_and_reap(&mut child);
            return Err(io::Error::new(io::ErrorKind::TimedOut, format!("getent timed out after {}ms", timeout.as_millis())));
        }
        loop {
            match stdout.read(&mut chunk) {
                Ok(0) => {
                    eof = true;
                    break;
                }
                Ok(n) => {
                    if out.len().saturating_add(n) > max_bytes {
                        drop(stdout);
                        kill_and_reap(&mut child);
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "getent output exceeded size limit"));
                    }
                    out.extend_from_slice(&chunk[..n]);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    drop(stdout);
                    kill_and_reap(&mut child);
                    return Err(e);
                }
            }
        }
    }
    drop(stdout);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(GetentRaw::Ran(status, out)),
            Ok(None) => {
                if Instant::now() >= deadline {
                    kill_and_reap(&mut child);
                    return Err(io::Error::new(io::ErrorKind::TimedOut, format!("getent timed out after {}ms", timeout.as_millis())));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => {
                kill_and_reap(&mut child);
                return Err(e);
            }
        }
    }
}

/// Runs `program -- <db> <key>`, returning `Ok(Some(stdout))` on exit 0,
/// `Ok(None)` on exit 2 (no such key) or missing binary, and `Err` otherwise.
fn run_getent_with_limits(program: &Path, db: &str, key: &[u8], timeout: Duration, max_bytes: usize) -> io::Result<Option<Vec<u8>>> {
    if !valid_key(key) {
        return Ok(None);
    }
    let args = [OsStr::new(db), OsStr::from_bytes(key)];
    match exec_getent(program, &args, timeout, max_bytes)? {
        GetentRaw::NoBinary => Ok(None),
        GetentRaw::Ran(status, stdout) => match status.code() {
            Some(0) => Ok(Some(stdout)),
            // 2: no such key.
            Some(2) => Ok(None),
            _ => Err(io::Error::other(format!("getent {db} {}: {status}", String::from_utf8_lossy(key)))),
        },
    }
}

/// An account the system knows from another source than /etc/passwd (SSSD,
/// LDAP, winbind, Entra ID, systemd-homed): this binary is static, so its own
/// lookups read /etc/passwd alone and can't load the system's NSS modules
/// (#349). `getent` uses them. Ok(None): no such account, or no getent.
fn getent_account(key: &[u8]) -> io::Result<Option<Account>> {
    let Some(program) = find_getent() else { return Ok(None) };
    getent_account_with(program, key)
}

fn getent_account_with(program: &Path, key: &[u8]) -> io::Result<Option<Account>> {
    getent_account_with_limits(program, key, GETENT_TIMEOUT, GETENT_MAX_STDOUT)
}

fn getent_account_with_limits(program: &Path, key: &[u8], timeout: Duration, max_bytes: usize) -> io::Result<Option<Account>> {
    let Some(stdout) = run_getent_with_limits(program, "passwd", key, timeout, max_bytes)? else { return Ok(None) };
    passwd_line(&stdout)
        .map(Some)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("unexpected getent passwd output for {}", String::from_utf8_lossy(key))))
}

/// A group resolved through NSS (`getent group`), returning `Ok(None)` only
/// when no `getent` binary exists so the caller can fall back to `/etc/group`.
fn getent_group_lookup(key: &[u8]) -> io::Result<Option<LookupResult<Group>>> {
    let Some(program) = find_getent() else { return Ok(None) };
    getent_group_lookup_with(program, key)
}

fn getent_group_lookup_with(program: &Path, key: &[u8]) -> io::Result<Option<LookupResult<Group>>> {
    if !valid_key(key) {
        return Ok(Some(LookupResult::NotFound));
    }
    let args = [OsStr::new("group"), OsStr::from_bytes(key)];
    match exec_getent(program, &args, GETENT_TIMEOUT, GETENT_MAX_STDOUT)? {
        GetentRaw::NoBinary => Ok(None),
        GetentRaw::Ran(status, stdout) => match status.code() {
            Some(0) => group_line(&stdout)
                .map(|g| Some(LookupResult::Found(g)))
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("unexpected getent group output for {}", String::from_utf8_lossy(key)))),
            Some(2) => Ok(Some(LookupResult::UnavailableOrNotFound)),
            _ => Err(io::Error::other(format!("getent group {}: {status}", String::from_utf8_lossy(key)))),
        },
    }
}

#[cfg(test)]
fn getent_group(key: &[u8]) -> io::Result<Option<Group>> {
    match getent_group_lookup(key)? {
        Some(LookupResult::Found(g)) => Ok(Some(g)),
        _ => Ok(None),
    }
}

#[cfg(test)]
fn getent_group_with(program: &Path, key: &[u8]) -> io::Result<Option<Group>> {
    match getent_group_lookup_with(program, key)? {
        Some(LookupResult::Found(g)) => Ok(Some(g)),
        _ => Ok(None),
    }
}

/// Checks supplementary group memberships through NSS (`getent initgroups`),
/// covering directory memberships that are not written into /etc/group.
fn initgroups_has(user: &CStr, gid: u32) -> io::Result<bool> {
    let Some(program) = find_getent() else { return Ok(false) };
    initgroups_has_with(program, user, gid)
}

fn initgroups_has_with(program: &Path, user: &CStr, gid: u32) -> io::Result<bool> {
    initgroups_has_with_limits(program, user, gid, GETENT_TIMEOUT, GETENT_MAX_STDOUT)
}

fn initgroups_has_with_limits(program: &Path, user: &CStr, gid: u32, timeout: Duration, max_bytes: usize) -> io::Result<bool> {
    let key = user.to_bytes();
    if !valid_key(key) {
        return Ok(false);
    }
    let args = [OsStr::new("initgroups"), OsStr::from_bytes(key)];
    match exec_getent(program, &args, timeout, max_bytes)? {
        GetentRaw::NoBinary => Ok(false),
        GetentRaw::Ran(status, stdout) => match status.code() {
            Some(0) => {
                let gids = initgroups_line(&stdout, key)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("unexpected getent initgroups output for {}", user.to_string_lossy())))?;
                Ok(gids.contains(&gid))
            }
            // 1: database unknown (non-glibc getent); 2: no such key.
            Some(1 | 2) => Ok(false),
            _ => Err(io::Error::other(format!("getent initgroups {}: {status}", user.to_string_lossy()))),
        },
    }
}

/// `name:password:uid:gid:gecos:dir:shell`, the single line of `getent passwd`.
fn passwd_line(text: &[u8]) -> Option<Account> {
    let line = single_line(text)?;
    let mut f = line.split(|&b| b == b':');
    let name = CString::new(f.next().filter(|n| !n.is_empty())?).ok()?;
    let _password = f.next()?;
    let uid = parse_u32(f.next()?)?;
    let gid = parse_u32(f.next()?)?;
    let _gecos = f.next()?;
    let _dir = f.next()?;
    let _shell = f.next()?;
    Some(Account { name, uid, gid })
}

/// `name:password:gid:members`, the single line of `getent group`.
fn group_line(text: &[u8]) -> Option<Group> {
    let line = single_line(text)?;
    let mut f = line.split(|&b| b == b':');
    let name = CString::new(f.next().filter(|n| !n.is_empty())?).ok()?;
    let _password = f.next()?;
    let gid = parse_u32(f.next()?)?;
    let raw_members = f.next()?;
    if f.next().is_some() {
        return None;
    }
    let mut members = Vec::new();
    for m in raw_members.split(|&b| b == b',').filter(|m| !m.is_empty()) {
        members.push(CString::new(m).ok()?);
    }
    Some(Group { name, gid, members })
}

/// `<user> [<gid> ...]`, the single line of `getent initgroups <user>`.
fn initgroups_line(text: &[u8], expected_user: &[u8]) -> Option<Vec<u32>> {
    let line = single_line(text)?;
    let rest = line.strip_prefix(expected_user)?;
    if !rest.is_empty() && !matches!(rest[0], b' ' | b'\t') {
        return None;
    }
    let mut gids = Vec::new();
    for tok in rest.split(|b| matches!(b, b' ' | b'\t')).filter(|s| !s.is_empty()) {
        gids.push(parse_initgroups_gid(tok)?);
    }
    Some(gids)
}

fn group_by_name_with(
    name: &CStr,
    getent_lookup: impl Fn(&[u8]) -> io::Result<Option<LookupResult<Group>>>,
    local_lookup: impl Fn(&CStr) -> io::Result<Option<Group>>,
) -> io::Result<LookupResult<Group>> {
    // Prefer `getent group` when available so /etc/nsswitch.conf ordering and
    // `[SUCCESS=merge]` are honored. Fall back to static `getgrnam_r` only
    // when no `getent` binary exists.
    match getent_lookup(name.to_bytes())? {
        Some(LookupResult::Found(g)) => Ok(match_group_name(name.to_bytes(), g).map_or(LookupResult::NotFound, LookupResult::Found)),
        Some(other) => Ok(other),
        None => Ok(local_lookup(name)?.map_or(LookupResult::NotFound, LookupResult::Found)),
    }
}

fn group_by_name_detailed(name: &CStr) -> io::Result<LookupResult<Group>> {
    group_by_name_with(name, getent_group_lookup, local_group_by_name)
}

fn group_by_name(name: &CStr) -> io::Result<Group> {
    match group_by_name_detailed(name)? {
        LookupResult::Found(g) => Ok(g),
        LookupResult::NotFound | LookupResult::UnavailableOrNotFound => Err(io::Error::new(io::ErrorKind::NotFound, format!("no group {}", name.to_string_lossy()))),
    }
}

fn c_name(s: &str) -> io::Result<CString> {
    CString::new(s).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))
}

fn group_members(group: &CStr) -> io::Result<Vec<String>> {
    let g = group_by_name(group)?;
    Ok(g.members.iter().map(|n| n.to_string_lossy().into_owned()).collect())
}

/// The users in `group` by name (its member list, not primary-group users).
pub fn members(group: &str) -> io::Result<Vec<String>> {
    group_members(&c_name(group)?)
}

/// A group's numeric id.
pub fn group_gid(group: &OsStr) -> io::Result<u32> {
    let name = CString::new(group.as_bytes()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    Ok(group_by_name(&name)?.gid)
}

/// A user's numeric id, if the account exists.
#[cfg(test)]
pub fn uid(user: &str) -> Option<u32> {
    Some(account_by_name(&c_name(user).ok()?).ok()??.uid)
}

/// Resolves an account by name and returns its canonical name from the user
/// database (which may differ in case or domain suffix for directory accounts).
pub fn account_name(user: &str) -> io::Result<Option<String>> {
    let Ok(name) = c_name(user) else { return Ok(None) };
    match account_by_name(&name)? {
        Some(a) => a
            .name
            .into_string()
            .map(Some)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("non-UTF-8 account name for {user}"))),
        None => Ok(None),
    }
}

/// The uids of the names in a group record, sorted. A name with no account
/// is left out; a lookup that fails fails the whole list, so a slow directory
/// never shrinks it.
fn uids_of(group: &Group) -> io::Result<Vec<u32>> {
    uids_of_with(group, account_by_name)
}

fn uids_of_with(group: &Group, lookup: impl Fn(&CStr) -> io::Result<Option<Account>>) -> io::Result<Vec<u32>> {
    let mut out = Vec::new();
    for name in &group.members {
        if let Some(a) = lookup(name)? {
            out.push(a.uid);
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn member_uids(group: &CStr) -> io::Result<Vec<u32>> {
    uids_of(&group_by_name(group)?)
}

/// Gives every member of `group` read-write access to `socket` by name, so a
/// member just added can use it without signing in again. Returns how many.
pub fn grant(socket: &Path, group: &str) -> io::Result<usize> {
    let uids = member_uids(&c_name(group)?)?;
    set(socket, &uids)?;
    Ok(uids.len())
}

/// The uid at the other end of an accepted connection.
fn peer_uid(conn: &UnixStream) -> io::Result<u32> {
    // SAFETY: a connected socket and zeroed storage for its credentials.
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: the output pointers are valid and the buffer is the stated size.
    let r = unsafe { libc::getsockopt(conn.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, (&mut cred as *mut libc::ucred).cast(), &mut len) };
    if r != 0 || len as usize != std::mem::size_of::<libc::ucred>() {
        return Err(if r != 0 { io::Error::last_os_error() } else { io::Error::new(io::ErrorKind::InvalidData, "short peer credentials") });
    }
    Ok(cred.uid)
}

/// Whether the account is in the group by name or as its primary group.
fn group_member(user: &CStr, primary_gid: libc::gid_t, group: &Group) -> bool {
    primary_gid == group.gid || group.members.iter().any(|m| m.as_c_str() == user)
}

/// The group's record, or `None` if the group was deleted: that leaves
/// nobody in it, which is an answer, not a failure.
fn current(group: &CStr) -> io::Result<Option<Group>> {
    match group_by_name(group) {
        Ok(g) => Ok(Some(g)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Whether the account `uid` is in the group now. No account, or no group: no.
fn member_of(uid: u32, group: Option<&Group>) -> io::Result<bool> {
    member_of_with(uid, group, account_by_uid, initgroups_has)
}

fn member_of_with(
    uid: u32,
    group: Option<&Group>,
    lookup: impl Fn(u32) -> io::Result<Option<Account>>,
    initgroups: impl Fn(&CStr, u32) -> io::Result<bool>,
) -> io::Result<bool> {
    let Some(group) = group else { return Ok(false) };
    let Some(a) = lookup(uid)? else { return Ok(false) };
    if group_member(&a.name, a.gid, group) {
        return Ok(true);
    }
    initgroups(&a.name, group.gid)
}

fn member(uid: u32, group: &CStr) -> io::Result<bool> {
    member_of(uid, current(group)?.as_ref())
}

/// Who may use the engine's socket: root, the account the daemon runs as, and
/// the members of one group, checked when they connect and again while they
/// stay connected.
pub struct Gate {
    group: CString,
    /// The daemon's socket, and the one containers are given (caller.rs).
    sockets: Vec<PathBuf>,
    owner: u32,
    open: Mutex<HashMap<u64, (u32, UnixStream)>>,
    next: AtomicU64,
    failing: AtomicBool,
    /// Recent answers by uid, so the docker CLI's burst of connections per
    /// command costs one lookup. They last no longer than the recheck does.
    recent: Mutex<HashMap<u32, (bool, Instant)>>,
}

/// A connection the gate let in; dropping it stops watching that connection.
pub struct Seat {
    gate: Arc<Gate>,
    id: u64,
}

impl Drop for Seat {
    fn drop(&mut self) {
        self.gate.open.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.id);
    }
}

impl Gate {
    pub fn new(sockets: &[&Path], group: &OsStr) -> io::Result<Arc<Gate>> {
        let group = CString::new(group.as_bytes()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: geteuid cannot fail.
        let owner = unsafe { libc::geteuid() };
        Ok(Arc::new(Gate { group, sockets: sockets.iter().map(|s| s.to_path_buf()).collect(), owner, open: Mutex::default(), next: AtomicU64::new(0), failing: AtomicBool::new(false), recent: Mutex::default() }))
    }

    fn name(&self) -> std::borrow::Cow<'_, str> {
        self.group.to_string_lossy()
    }

    fn allowed(&self, uid: u32) -> io::Result<bool> {
        if uid == 0 || uid == self.owner {
            return Ok(true);
        }
        if let Some((yes, at)) = self.recent.lock().unwrap_or_else(|e| e.into_inner()).get(&uid)
            && at.elapsed() < RECHECK
        {
            return Ok(*yes);
        }
        let yes = member(uid, &self.group)?;
        self.remember(uid, yes);
        Ok(yes)
    }

    fn remember(&self, uid: u32, yes: bool) {
        let mut recent = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        recent.retain(|_, (_, at)| at.elapsed() < RECHECK);
        recent.insert(uid, (yes, Instant::now()));
    }

    /// One recheck: everyone connected against one read of the group, then
    /// the access list, if the group's names changed since it was written.
    /// `listed`: the names the list was last written for.
    fn tick(&self, listed: &mut Option<Vec<CString>>) {
        let group = current(&self.group);
        let connected = !self.open.lock().unwrap_or_else(|e| e.into_inner()).is_empty();
        if connected {
            let closed = self.sweep_with(|uid| {
                if uid == 0 || uid == self.owner {
                    return Ok(true);
                }
                let g = group.as_ref().map_err(|e| io::Error::new(e.kind(), e.to_string()))?;
                let yes = member_of(uid, g.as_ref())?;
                self.remember(uid, yes);
                Ok(yes)
            });
            if closed > 0 {
                *listed = None;
            }
        }
        let Ok(group) = group else { return };
        let now: Vec<CString> = group.as_ref().map(|g| g.members.clone()).unwrap_or_default();
        if listed.as_ref() == Some(&now) {
            return;
        }
        // Only a list actually written counts: anything else is tried again.
        let written = group.as_ref().map_or(Ok(Vec::new()), uids_of).and_then(|uids| self.set_all(&uids).map(|()| uids.len()));
        match written {
            Ok(n) => {
                crate::diag::info!("socket access list written", members = n);
                *listed = Some(now);
            }
            Err(e) if unsupported(&e) => *listed = Some(now),
            Err(e) => crate::diag::warn!("socket access list not written", error = e),
        }
    }

    /// Writes the socket's access list for the group's current members.
    pub fn grant(&self) -> io::Result<usize> {
        let uids = member_uids(&self.group)?;
        self.set_all(&uids)?;
        Ok(uids.len())
    }

    /// Writes each socket's access list.
    fn set_all(&self, uids: &[u32]) -> io::Result<()> {
        set_each(&self.sockets, uids, set)
    }

    /// Checks a new connection. `None`: refused, and the reason logged.
    pub fn admit(self: &Arc<Self>, conn: &UnixStream) -> Option<Seat> {
        self.admit_with(conn, |uid| self.allowed(uid))
    }

    fn admit_with(self: &Arc<Self>, conn: &UnixStream, check: impl Fn(u32) -> io::Result<bool>) -> Option<Seat> {
        let checked = peer_uid(conn).and_then(|uid| Ok((uid, check(uid)?)));
        let uid = match checked {
            Ok((uid, true)) => uid,
            Ok((uid, false)) => {
                crate::diag::warn!("socket connection refused: user is not in the access group", group = self.name(), uid = uid);
                return None;
            }
            Err(e) => {
                crate::diag::warn!("socket connection refused: access check failed", group = self.name(), error = e);
                return None;
            }
        };
        let keep = match conn.try_clone() {
            Ok(c) => c,
            Err(e) => {
                crate::diag::warn!("socket connection refused: cannot watch it", error = e);
                return None;
            }
        };
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.open.lock().unwrap_or_else(|e| e.into_inner()).insert(id, (uid, keep));
        Some(Seat { gate: self.clone(), id })
    }

    /// Checks everyone still connected, once per account, and closes the
    /// connections of anyone no longer allowed. A failed lookup leaves the
    /// connections open (new ones are refused until lookups work again).
    fn sweep_with(&self, check: impl Fn(u32) -> io::Result<bool>) -> usize {
        let uids: HashSet<u32> = self.open.lock().unwrap_or_else(|e| e.into_inner()).values().map(|(u, _)| *u).collect();
        let mut removed = HashSet::new();
        let mut failed = None;
        for uid in uids {
            match check(uid) {
                Ok(true) => {}
                Ok(false) => {
                    removed.insert(uid);
                }
                Err(e) => failed = Some(e),
            }
        }
        match failed {
            Some(e) if !self.failing.swap(true, Ordering::Relaxed) => crate::diag::warn!("access check failed; open connections kept until it works again", group = self.name(), error = e),
            None if self.failing.swap(false, Ordering::Relaxed) => crate::diag::info!("access check works again", group = self.name()),
            _ => {}
        }
        if removed.is_empty() {
            return 0;
        }
        let mut closed = 0;
        // Taken out as they're closed, so a thread that takes a while to
        // notice isn't closed and logged again on every recheck.
        self.open.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, (uid, conn)| {
            if !removed.contains(uid) {
                return true;
            }
            // Ends the connection for every thread using it: reads see the
            // end, writes fail, as if the client had hung up.
            let _ = conn.shutdown(Shutdown::Both);
            closed += 1;
            false
        });
        for uid in &removed {
            crate::diag::warn!("connections closed: user is no longer in the access group", group = self.name(), uid = uid);
        }
        closed
    }

    /// Checks everyone connected every few seconds, and writes the access
    /// list again whenever the group's members change.
    pub fn watch(self: Arc<Self>) -> io::Result<()> {
        std::thread::Builder::new().name("kiri-access".into()).spawn(move || {
            // Unknown at first: the first recheck writes the list once more,
            // so a change between startup's write and now isn't missed.
            let mut listed = None;
            loop {
                std::thread::sleep(RECHECK);
                self.tick(&mut listed);
            }
        })?;
        Ok(())
    }
}

/// Writes the socket's access list: its owner and group (rw, as the mode
/// says), each of `uids` (rw), nobody else.
pub fn set(socket: &Path, uids: &[u32]) -> io::Result<()> {
    let blob = acl(uids);
    let path = CString::new(socket.as_os_str().as_bytes()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: both strings are valid and NUL-terminated; blob outlives the call.
    let r = unsafe { libc::setxattr(path.as_ptr(), c"system.posix_acl_access".as_ptr(), blob.as_ptr().cast(), blob.len(), 0) };
    if r == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}

/// Writes the access list of every one of `sockets` with `set`, each tried
/// whatever became of the one before; the first error, if any.
fn set_each(sockets: &[PathBuf], uids: &[u32], set: impl Fn(&Path, &[u32]) -> io::Result<()>) -> io::Result<()> {
    let mut first = Ok(());
    for s in sockets {
        if let Err(e) = set(s, uids) {
            first = first.and(Err(e));
        }
    }
    first
}

/// Whether a failure to write the list means this system can't keep one
/// (a filesystem without ACLs): then only signing in again lets a new member in.
pub fn unsupported(e: &io::Error) -> bool {
    e.raw_os_error() == Some(libc::EOPNOTSUPP)
}

/// The kernel's form of an access list (`system.posix_acl_access`): a version,
/// then entries of tag, permissions and id, in tag order and, within a tag,
/// in id order.
fn acl(uids: &[u32]) -> Vec<u8> {
    const USER_OBJ: u16 = 0x01;
    const USER: u16 = 0x02;
    const GROUP_OBJ: u16 = 0x04;
    const MASK: u16 = 0x10;
    const OTHER: u16 = 0x20;
    const NO_ID: u32 = u32::MAX;
    const RW: u16 = 6;
    fn entry(out: &mut Vec<u8>, tag: u16, perm: u16, id: u32) {
        out.extend(tag.to_le_bytes());
        out.extend(perm.to_le_bytes());
        out.extend(id.to_le_bytes());
    }
    let mut ids: Vec<u32> = uids.iter().copied().filter(|&u| u != 0).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut out = 2u32.to_le_bytes().to_vec();
    entry(&mut out, USER_OBJ, RW, NO_ID);
    for id in ids {
        entry(&mut out, USER, RW, id);
    }
    entry(&mut out, GROUP_OBJ, RW, NO_ID);
    entry(&mut out, MASK, RW, NO_ID);
    entry(&mut out, OTHER, 0, NO_ID);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One socket's list that cannot be written does not keep the next
    /// from being written; the error is still reported.
    #[test]
    fn every_sockets_list_is_written_when_one_fails() {
        let sockets = [PathBuf::from("/run/a.sock"), PathBuf::from("/run/b.sock")];
        let written = Mutex::new(Vec::new());
        let r = set_each(&sockets, &[1000], |s, _| {
            written.lock().unwrap().push(s.to_path_buf());
            if s.ends_with("a.sock") { Err(io::Error::other("no")) } else { Ok(()) }
        });
        assert!(r.is_err());
        assert_eq!(*written.lock().unwrap(), sockets);
    }

    #[test]
    fn acl_lists_each_user_once_in_order_and_not_root() {
        let b = acl(&[1001, 0, 1000, 1001]);
        assert_eq!(&b[..4], &[2, 0, 0, 0]);
        let entries: Vec<(u16, u16, u32)> = b[4..]
            .chunks(8)
            .map(|c| (u16::from_le_bytes([c[0], c[1]]), u16::from_le_bytes([c[2], c[3]]), u32::from_le_bytes([c[4], c[5], c[6], c[7]])))
            .collect();
        assert_eq!(entries, vec![(1, 6, u32::MAX), (2, 6, 1000), (2, 6, 1001), (4, 6, u32::MAX), (0x10, 6, u32::MAX), (0x20, 0, u32::MAX)]);
    }

    #[test]
    fn membership_includes_names_and_primary_group() {
        let group = Group { name: c"kiri".to_owned(), gid: 100, members: vec![c"alice".to_owned()] };
        assert!(group_member(c"alice", 1, &group));
        assert!(group_member(c"bob", 100, &group));
        assert!(!group_member(c"bob", 1, &group));
    }

    #[test]
    fn accepted_connection_reports_its_peer_uid() {
        let (a, _b) = UnixStream::pair().unwrap();
        // SAFETY: getuid cannot fail.
        assert_eq!(peer_uid(&a).unwrap(), unsafe { libc::getuid() });
    }

    /// The real lookup, without the root and owner shortcuts. Skipped where
    /// the test's uid has no account (some CI containers).
    #[test]
    fn current_users_primary_group_counts() {
        // SAFETY: getuid cannot fail.
        let uid = unsafe { libc::getuid() };
        let Some(a) = account_by_uid(uid).unwrap() else { return };
        let g = {
            // SAFETY: getgrgid_r with a group record and a buffer of the stated size.
            let gid = a.gid;
            unsafe { nss(|rec, buf, len, out| libc::getgrgid_r(gid, rec, buf, len, out)) }.unwrap()
        };
        let Some(g) = g else { return };
        // SAFETY: gr_name is a NUL-terminated string in the record's buffer.
        let name = unsafe { CStr::from_ptr(g.rec.gr_name) }.to_owned();
        assert!(member(uid, &name).unwrap());
        assert!(!member(u32::MAX - 7, &name).unwrap(), "no such account");
        assert!(!member(uid, c"kiri-no-such-group-x").unwrap(), "a deleted group has nobody in it");
    }

    #[test]
    fn lookups_from_many_threads_agree() {
        // SAFETY: getuid cannot fail.
        let uid = unsafe { libc::getuid() };
        let Some(me) = account_by_uid(uid).unwrap() else { return };
        // Asked of getent too, so once rather than in the loop.
        assert!(account_by_uid(u32::MAX - 7).unwrap().is_none());
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..200 {
                        assert_eq!(account_by_uid(uid).unwrap().unwrap(), me);
                    }
                });
            }
        });
    }

    fn gate() -> Arc<Gate> {
        let mut g = Gate::new(&[Path::new("/nonexistent")], OsStr::new("kiri")).unwrap();
        Arc::get_mut(&mut g).unwrap().owner = u32::MAX;
        g
    }

    #[test]
    fn root_and_the_daemons_own_account_need_no_group() {
        let g = gate();
        assert!(g.allowed(0).unwrap());
        let mut g2 = Gate::new(&[Path::new("/nonexistent")], OsStr::new("kiri-no-such-group-x")).unwrap();
        Arc::get_mut(&mut g2).unwrap().owner = 4242;
        assert!(g2.allowed(4242).unwrap());
        assert!(!g2.allowed(4243).unwrap(), "anyone else is looked up");
    }

    #[test]
    fn refused_or_failed_checks_keep_the_connection_out() {
        let g = gate();
        let (a, _b) = UnixStream::pair().unwrap();
        assert!(g.admit_with(&a, |_| Ok(false)).is_none());
        assert!(g.admit_with(&a, |_| Err(io::Error::other("nss down"))).is_none());
        assert!(g.open.lock().unwrap().is_empty());
        let seat = g.admit_with(&a, |_| Ok(true)).unwrap();
        assert_eq!(g.open.lock().unwrap().len(), 1);
        drop(seat);
        assert!(g.open.lock().unwrap().is_empty(), "a closed connection isn't watched");
    }

    #[test]
    fn removal_closes_an_open_connection_mid_stream() {
        use std::io::{Read, Write};
        let g = gate();
        let (a, mut b) = UnixStream::pair().unwrap();
        let _seat = g.admit_with(&a, |_| Ok(true)).unwrap();
        // A thread blocked reading the client, as an exec session's stdin is.
        let mut reader = a.try_clone().unwrap();
        let blocked = std::thread::spawn(move || reader.read(&mut [0; 16]).unwrap());
        b.write_all(b"x").unwrap();
        // Still a member: nothing happens.
        assert_eq!(g.sweep_with(|_| Ok(true)), 0);
        // A lookup failure keeps it open too.
        assert_eq!(g.sweep_with(|_| Err(io::Error::other("nss down"))), 0);
        assert_eq!(blocked.join().unwrap(), 1);
        let mut reader = a.try_clone().unwrap();
        let blocked = std::thread::spawn(move || reader.read(&mut [0; 16]).unwrap());
        std::thread::sleep(Duration::from_millis(50));
        // Removed: the blocked read sees the end, and the client does too.
        assert_eq!(g.sweep_with(|_| Ok(false)), 1);
        assert_eq!(g.sweep_with(|_| Ok(false)), 0, "closed once, not again on every recheck");
        assert_eq!(blocked.join().unwrap(), 0);
        assert_eq!(b.read(&mut [0; 16]).unwrap(), 0);
        assert!((&a).write_all(b"y").is_err());
    }

    /// The kernel takes the list and lets a user who isn't in the socket's
    /// group in by it, at once. Needs root (cargo test -- --ignored, as root).
    #[test]
    #[ignore]
    fn a_listed_user_gets_in_and_others_dont() {
        use std::os::unix::fs::PermissionsExt;
        let dir = Path::new("/dev/shm").join(format!("kiri-acl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let sock = dir.join("s.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o660)).unwrap();
        // As uid `as_uid` (no groups): can it connect?
        let try_as = |as_uid: u32| {
            // SAFETY: fork in a test; the child only drops ids, connects and exits.
            unsafe {
                let pid = libc::fork();
                if pid == 0 {
                    libc::setgroups(0, std::ptr::null());
                    libc::setgid(as_uid);
                    libc::setuid(as_uid);
                    libc::_exit(if std::os::unix::net::UnixStream::connect(&sock).is_ok() { 0 } else { 1 });
                }
                let mut status = 0;
                libc::waitpid(pid, &mut status, 0);
                libc::WEXITSTATUS(status) == 0
            }
        };
        assert!(!try_as(65534), "nobody isn't let in before the list");
        set(&sock, &[65534]).unwrap();
        assert!(try_as(65534), "a listed user gets in");
        assert!(!try_as(65533), "an unlisted one doesn't");
        set(&sock, &[]).unwrap();
        assert!(!try_as(65534), "removing the entry takes access away");

        // An old ACL entry is not enough: the daemon's group check still says no.
        set(&sock, &[65534]).unwrap();
        assert!(try_as(65534));
        let (conn, _) = listener.accept().unwrap();
        let mut gate = Gate::new(&[sock.as_path()], OsStr::new("root")).unwrap();
        Arc::get_mut(&mut gate).unwrap().owner = u32::MAX;
        assert!(gate.admit(&conn).is_none(), "nobody is not in root's group");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A real account, added to a real group and taken out again while
    /// connected: its open connection is closed and the list rewritten.
    /// Needs root and useradd (cargo test -- --ignored, as root, in a throwaway system).
    #[test]
    #[ignore]
    fn removing_a_real_member_closes_their_connection() {
        use std::io::Read;
        use std::os::unix::fs::PermissionsExt;
        use std::process::Command;
        let run = |args: &[&str]| assert!(Command::new(args[0]).args(&args[1..]).status().unwrap().success(), "{args:?}");
        let _ = Command::new("userdel").arg("kiritest").status();
        let _ = Command::new("groupdel").arg("kiritestg").status();
        run(&["groupadd", "kiritestg"]);
        run(&["useradd", "-M", "-G", "kiritestg", "kiritest"]);
        let who = uid("kiritest").unwrap();
        assert!(members("kiritestg").unwrap().contains(&"kiritest".to_owned()));

        let dir = Path::new("/dev/shm").join(format!("kiri-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let sock = dir.join("s.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o660)).unwrap();
        let mut gate = Gate::new(&[sock.as_path()], OsStr::new("kiritestg")).unwrap();
        Arc::get_mut(&mut gate).unwrap().owner = u32::MAX;
        assert_eq!(gate.grant().unwrap(), 1);

        // The member connects (by the list: they have no group in this
        // process) and waits for an answer that never comes.
        // SAFETY: fork in a test; the child only drops ids, connects, reads and exits.
        let pid = unsafe {
            let pid = libc::fork();
            if pid == 0 {
                // A gid that isn't the test group, so only the list can let it in.
                if libc::setgroups(0, std::ptr::null()) != 0 || libc::setgid(65534) != 0 || libc::setuid(who) != 0 {
                    libc::_exit(3);
                }
                let code = match std::os::unix::net::UnixStream::connect(&sock) {
                    Ok(mut c) => match c.read(&mut [0; 1]) {
                        Ok(0) => 0,
                        _ => 2,
                    },
                    Err(_) => 1,
                };
                libc::_exit(code);
            }
            pid
        };
        let (conn, _) = listener.accept().unwrap();
        let _seat = gate.admit(&conn).expect("a member is let in");
        let mut listed = None;
        gate.tick(&mut listed);
        assert_eq!(gate.open.lock().unwrap().len(), 1, "still a member");

        run(&["gpasswd", "-d", "kiritest", "kiritestg"]);
        gate.tick(&mut listed);
        assert!(gate.open.lock().unwrap().is_empty(), "removed: their connection is closed");
        assert_eq!(listed, Some(vec![]), "and the list is written without them");
        let mut status = 0;
        // SAFETY: waits for the child forked above.
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(libc::WEXITSTATUS(status), 0, "the client saw the connection end");
        assert_eq!(gate.grant().unwrap(), 0, "and the list no longer names them");

        let _ = Command::new("userdel").arg("kiritest").status();
        let _ = Command::new("groupdel").arg("kiritestg").status();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn acl_without_users_is_the_plain_mode() {
        assert_eq!(acl(&[]).len(), 4 + 4 * 8);
    }

    /// Accounts only NSS knows (#349) come from getent's passwd line.
    #[test]
    fn getent_lines_read_as_accounts() {
        let alice = Account { name: c"alice".to_owned(), uid: 5001, gid: 5001 };
        assert_eq!(passwd_line(b"alice:x:5001:5001:Directory user:/home/alice:/bin/bash\n"), Some(alice));
        assert_eq!(passwd_line(b"alice@corp.example:*:1916400001:1916400001::/home/alice:/bin/sh").map(|a| a.uid), Some(1_916_400_001));
        assert_eq!(passwd_line(b"CORP\\alice:x:5002:5002:Room: 101:/home/alice:/bin/sh").map(|a| a.uid), Some(5002));
        for bad in [
            &b""[..],
            b"\n",
            b":x:1:1::/:/bin/sh",
            b"alice:x:one:1::/:/bin/sh",
            b"alice:x:+5001:5001::/:/bin/sh",
            b"alice:x:1",
            b"alice:x:1:1",
            b"alice:x:5001:5001::/:/bin/sh\nbob:x:5002:5002::/:/bin/sh\n",
        ] {
            assert_eq!(passwd_line(bad), None, "{bad:?}");
        }
        // Never an option or delimiter to getent.
        for bad_key in [&b""[..], b"--help", b"-s", b"alice:x", b"alice\nbob"] {
            assert_eq!(getent_account(bad_key).unwrap(), None, "{bad_key:?}");
            assert_eq!(getent_group(bad_key).unwrap(), None, "{bad_key:?}");
        }
    }

    #[test]
    fn getent_lines_read_as_groups() {
        let g = Group { name: c"kiri".to_owned(), gid: 998, members: vec![c"alice".to_owned(), c"bob@corp.example".to_owned()] };
        assert_eq!(group_line(b"kiri:x:998:alice,bob@corp.example\n"), Some(g));
        assert_eq!(group_line(b"kiri:x:998:\n"), Some(Group { name: c"kiri".to_owned(), gid: 998, members: vec![] }));
        for bad in [
            &b""[..],
            b"\n",
            b":x:998:alice",
            b"kiri:x:abc:alice",
            b"kiri:x:+998:alice",
            b"kiri:x:998",
            b"kiri:x:998:alice:extra",
            b"kiri:x:998:alice\nother:x:999:bob\n",
        ] {
            assert_eq!(group_line(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn getent_initgroups_lines_match_exact_user_and_gids() {
        assert_eq!(initgroups_line(b"alice                 65200 1002\n", b"alice"), Some(vec![65200, 1002]));
        assert_eq!(initgroups_line(b"alice\t998\n", b"alice"), Some(vec![998]));
        assert_eq!(initgroups_line(b"alice                 \n", b"alice"), Some(vec![]));
        assert_eq!(initgroups_line(b"alice\n", b"alice"), Some(vec![]));
        assert_eq!(initgroups_line(b"alice@corp.example 998 44\n", b"alice@corp.example"), Some(vec![998, 44]));
        for bad in [
            &b""[..],
            b"\n",
            b"alice2 998\n",
            b"bob 998\n",
            b"alice 998 bad\n",
            b"alice +998\n",
            b"alice 998\nbob 999\n",
        ] {
            assert_eq!(initgroups_line(bad, b"alice"), None, "{bad:?}");
        }
    }

    #[test]
    fn getent_process_handles_status_codes_and_invalid_keys() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("kiri-getent-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let prog = dir.join("fake-getent");
        let touched = dir.join("touched");
        let script = format!(
            "#!/bin/sh\n\
             [ \"$1\" = \"--\" ] || exit 9\n\
             shift\n\
             db=\"$1\"; key=\"$2\"\n\
             case \"$key\" in -*) echo bad > \"{}\"; exit 9 ;; esac\n\
             case \"$db:$key\" in\n\
               passwd:alice) echo 'alice:x:5001:5001:Alice:/home/alice:/bin/sh'; exit 0 ;;\n\
               passwd:5001) echo 'alice:x:5001:5001:Alice:/home/alice:/bin/sh'; exit 0 ;;\n\
               passwd:0) echo 'root:x:0:0:root:/root:/bin/sh'; exit 0 ;;\n\
               passwd:badline) echo 'not-a-passwd-line'; exit 0 ;;\n\
               passwd:err) exit 3 ;;\n\
               group:dirkiri) echo 'dirkiri:x:65200:alice,localbob'; exit 0 ;;\n\
               group:badline) echo 'bad:group'; exit 0 ;;\n\
               group:err) exit 1 ;;\n\
               initgroups:alice) echo 'alice                 65200 1002'; exit 0 ;;\n\
               initgroups:outsider) echo 'outsider              2000'; exit 0 ;;\n\
               initgroups:unsupported) exit 1 ;;\n\
               initgroups:badline) echo 'alice2 1002'; exit 0 ;;\n\
               initgroups:err) exit 3 ;;\n\
               *) exit 2 ;;\n\
             esac\n",
            touched.display()
        );
        std::fs::write(&prog, script).unwrap();
        std::fs::set_permissions(&prog, std::fs::Permissions::from_mode(0o755)).unwrap();

        let alice = Account { name: c"alice".to_owned(), uid: 5001, gid: 5001 };
        assert_eq!(getent_account_with(&prog, b"alice").unwrap(), Some(alice));
        assert_eq!(getent_account_with(&prog, b"missing").unwrap(), None);
        assert_eq!(getent_account_with(&prog, b"badline").unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(getent_account_with(&prog, b"err").is_err());
        // Numeric username returning a different account name is rejected by the caller's exact check.
        assert_eq!(getent_account_with(&prog, b"0").unwrap().filter(|a| a.name.as_c_str() == c"0"), None);

        let dirkiri = Group { name: c"dirkiri".to_owned(), gid: 65200, members: vec![c"alice".to_owned(), c"localbob".to_owned()] };
        assert_eq!(getent_group_with(&prog, b"dirkiri").unwrap(), Some(dirkiri));
        assert_eq!(getent_group_with(&prog, b"missing").unwrap(), None);
        assert_eq!(getent_group_with(&prog, b"badline").unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(getent_group_with(&prog, b"err").is_err());

        assert!(initgroups_has_with(&prog, c"alice", 1002).unwrap());
        assert!(!initgroups_has_with(&prog, c"alice", 9999).unwrap());
        assert!(!initgroups_has_with(&prog, c"outsider", 1002).unwrap());
        assert!(!initgroups_has_with(&prog, c"unsupported", 1002).unwrap());
        assert!(!initgroups_has_with(&prog, c"missing", 1002).unwrap());
        assert_eq!(initgroups_has_with(&prog, c"badline", 1002).unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(initgroups_has_with(&prog, c"err", 1002).is_err());

        // Option-like keys never invoke the helper.
        assert_eq!(getent_account_with(&prog, b"-s").unwrap(), None);
        assert_eq!(getent_group_with(&prog, b"--help").unwrap(), None);
        assert!(!initgroups_has_with(&prog, c"-s", 1002).unwrap());
        assert!(!touched.exists());

        // Missing binary counts as no entry rather than an error.
        let missing = dir.join("no-such-getent");
        assert_eq!(getent_account_with(&missing, b"alice").unwrap(), None);
        assert_eq!(getent_group_with(&missing, b"dirkiri").unwrap(), None);
        assert!(!initgroups_has_with(&missing, c"alice", 1002).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn member_and_uids_enforce_configured_group_only() {
        let group = Group { name: c"kiri".to_owned(), gid: 998, members: vec![c"local".to_owned(), c"alice".to_owned(), c"ghost".to_owned()] };
        let accounts = |uid: u32| -> io::Result<Option<Account>> {
            Ok(match uid {
                1001 => Some(Account { name: c"local".to_owned(), uid: 1001, gid: 1001 }),
                5001 => Some(Account { name: c"alice".to_owned(), uid: 5001, gid: 5001 }),
                5002 => Some(Account { name: c"dirgroup".to_owned(), uid: 5002, gid: 5002 }),
                5003 => Some(Account { name: c"outsider".to_owned(), uid: 5003, gid: 5003 }),
                5004 => Some(Account { name: c"primary".to_owned(), uid: 5004, gid: 998 }),
                _ => None,
            })
        };
        let initgroups = |user: &CStr, gid: u32| -> io::Result<bool> { Ok(user == c"dirgroup" && gid == 998) };

        assert!(member_of_with(1001, Some(&group), accounts, initgroups).unwrap());
        assert!(member_of_with(5001, Some(&group), accounts, initgroups).unwrap());
        assert!(member_of_with(5002, Some(&group), accounts, initgroups).unwrap());
        assert!(member_of_with(5004, Some(&group), accounts, initgroups).unwrap());
        assert!(!member_of_with(5003, Some(&group), accounts, initgroups).unwrap(), "non-member refused");
        assert!(!member_of_with(7777, Some(&group), accounts, initgroups).unwrap(), "unknown uid refused");
        assert!(!member_of_with(5001, None, accounts, initgroups).unwrap(), "deleted group admits nobody");

        // uids_of resolves directory members, omits missing accounts, and fails closed on errors.
        let by_name = |name: &CStr| -> io::Result<Option<Account>> {
            Ok(match name.to_bytes() {
                b"local" => Some(Account { name: c"local".to_owned(), uid: 1001, gid: 1001 }),
                b"alice" => Some(Account { name: c"alice".to_owned(), uid: 5001, gid: 5001 }),
                _ => None,
            })
        };
        assert_eq!(uids_of_with(&group, by_name).unwrap(), vec![1001, 5001]);
        assert!(uids_of_with(&group, |_| Err(io::Error::other("directory unavailable"))).is_err());
    }

    fn write_test_script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn make_temp_dir(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("kiri-access-{tag}-{}-{}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn getent_runs_with_clean_environment() {
        let dir = make_temp_dir("env");
        let env_passwd = dir.join("env-passwd");
        let env_init = dir.join("env-init");
        let prog = write_test_script(
            &dir,
            "fake-getent",
            &format!(
                "#!/bin/sh\n\
                 [ \"$1\" = \"--\" ] || exit 9\n\
                 shift\n\
                 case \"$1\" in\n\
                   passwd) export -p > \"{}\"; echo 'alice:x:5001:5001::/:/bin/sh'; exit 0 ;;\n\
                   initgroups) export -p > \"{}\"; echo 'alice 998'; exit 0 ;;\n\
                   *) exit 2 ;;\n\
                 esac\n",
                env_passwd.display(),
                env_init.display(),
            ),
        );

        assert!(getent_account_with(&prog, b"alice").unwrap().is_some());
        assert!(initgroups_has_with(&prog, c"alice", 998).unwrap());

        for dump in [&env_passwd, &env_init] {
            let text = std::fs::read_to_string(dump).unwrap();
            let mut saw_lc_all = false;
            for line in text.lines() {
                let rest = line.strip_prefix("export ").unwrap_or(line);
                let (k, v) = rest.split_once('=').unwrap_or((rest, ""));
                if k == "LC_ALL" {
                    assert!(v == "'C'" || v == "C" || v == "\"C\"", "unexpected LC_ALL: {line}");
                    saw_lc_all = true;
                } else {
                    assert!(matches!(k, "PWD" | "SHLVL" | "_"), "unexpected inherited env var in getent: {line}");
                }
            }
            assert!(saw_lc_all, "LC_ALL=C not set in {text}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn getent_times_out_and_caps_stdout() {
        let dir = make_temp_dir("limits");
        let prog = write_test_script(
            &dir,
            "fake-getent",
            "#!/bin/sh\n\
             [ \"$1\" = \"--\" ] || exit 9\n\
             shift\n\
             case \"$2\" in\n\
               hang) [ -x /bin/sleep ] && /bin/sleep 30 || /usr/bin/sleep 30; exit 0 ;;\n\
               hang_closed) exec >&-; [ -x /bin/sleep ] && /bin/sleep 30 || /usr/bin/sleep 30; exit 0 ;;\n\
               flood)\n\
                 line='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n\
                 i=0; while [ \"$i\" -lt 64 ]; do printf '%s\\n' \"$line\"; i=$((i+1)); done\n\
                 exit 0 ;;\n\
               *) exit 2 ;;\n\
             esac\n",
        );

        let limit = Duration::from_millis(150);
        let t0 = Instant::now();
        let err = getent_account_with_limits(&prog, b"hang", limit, GETENT_MAX_STDOUT).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(t0.elapsed() < Duration::from_secs(3), "hung passwd lookup took {:?}", t0.elapsed());

        let t1 = Instant::now();
        let err = initgroups_has_with_limits(&prog, c"hang", 998, limit, GETENT_MAX_STDOUT).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(t1.elapsed() < Duration::from_secs(3), "hung initgroups lookup took {:?}", t1.elapsed());

        let t2 = Instant::now();
        let err = getent_account_with_limits(&prog, b"hang_closed", limit, GETENT_MAX_STDOUT).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(t2.elapsed() < Duration::from_secs(3), "closed-stdout hang took {:?}", t2.elapsed());

        let err = getent_account_with_limits(&prog, b"flood", Duration::from_secs(3), 256).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn group_by_name_prefers_getent_for_nss_merge_and_override() {
        let local = Group { name: c"kiri".to_owned(), gid: 100, members: vec![c"local".to_owned()] };
        let merged = Group { name: c"kiri".to_owned(), gid: 100, members: vec![c"local".to_owned(), c"diralice".to_owned()] };
        let overridden = Group { name: c"kiri".to_owned(), gid: 200, members: vec![c"sssonly".to_owned()] };

        // Merged local + NSS members are preserved when getent is available.
        let g = group_by_name_with(c"kiri", |_| Ok(Some(LookupResult::Found(merged.clone()))), |_| Ok(Some(local.clone()))).unwrap();
        assert_eq!(g, LookupResult::Found(merged));

        // NSS-first GID and membership override /etc/group when getent is available.
        let g = group_by_name_with(c"kiri", |_| Ok(Some(LookupResult::Found(overridden.clone()))), |_| Ok(Some(local.clone()))).unwrap();
        assert_eq!(g, LookupResult::Found(overridden));

        // Exit 2 from getent does not silently override NSS policy with /etc/group.
        let g = group_by_name_with(c"kiri", |_| Ok(Some(LookupResult::UnavailableOrNotFound)), |_| Ok(Some(local.clone()))).unwrap();
        assert_eq!(g, LookupResult::UnavailableOrNotFound);

        // Only when no getent binary exists do we fall back to static getgrnam_r.
        let g = group_by_name_with(c"kiri", |_| Ok(None), |_| Ok(Some(local.clone()))).unwrap();
        assert_eq!(g, LookupResult::Found(local));
    }

    #[test]
    fn canonical_directory_names_accepted_while_numeric_keys_require_exact_match() {
        let dir = make_temp_dir("canon");
        let prog = write_test_script(
            &dir,
            "fake-getent",
            "#!/bin/sh\n\
             [ \"$1\" = \"--\" ] || exit 9\n\
             shift\n\
             case \"$1:$2\" in\n\
               passwd:alice|passwd:Alice@corp.example) echo 'alice@corp.example:x:5001:5001::/home/alice:/bin/sh'; exit 0 ;;\n\
               passwd:0|'passwd:+0'|'passwd: 0'|'passwd: -0') echo 'root:x:0:0:root:/root:/bin/sh'; exit 0 ;;\n\
               group:DevOps) echo 'devops@corp.example:x:65300:alice@corp.example'; exit 0 ;;\n\
               group:0|'group:+0'|'group: 0') echo 'root:x:0:'; exit 0 ;;\n\
               *) exit 2 ;;\n\
             esac\n",
        );

        let canonical = Account { name: c"alice@corp.example".to_owned(), uid: 5001, gid: 5001 };
        for query in [&b"alice"[..], b"Alice@corp.example"] {
            let got = getent_account_with(&prog, query).unwrap().and_then(|a| match_account_name(query, a));
            assert_eq!(got, Some(canonical.clone()), "{query:?}");
        }
        for numeric in [&b"0"[..], b"+0", b" 0", b" -0"] {
            assert!(is_numeric_key(numeric), "{numeric:?}");
            let got = getent_account_with(&prog, numeric).unwrap().and_then(|a| match_account_name(numeric, a));
            assert_eq!(got, None, "{numeric:?} must not resolve to root");
        }
        assert!(is_numeric_key(b"\t+42"));
        assert!(!is_numeric_key(b"123alice"));
        assert!(!is_numeric_key(b"   "));

        let devops = Group { name: c"devops@corp.example".to_owned(), gid: 65300, members: vec![c"alice@corp.example".to_owned()] };
        let g = group_by_name_with(c"DevOps", |k| getent_group_lookup_with(&prog, k), |_| Ok(None)).unwrap();
        assert_eq!(g, LookupResult::Found(devops));
        for numeric in [c"0", c"+0", c" 0"] {
            let g = group_by_name_with(numeric, |k| getent_group_lookup_with(&prog, k), |_| Ok(None)).unwrap();
            assert_eq!(g, LookupResult::NotFound, "{numeric:?} must not resolve to group root");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn initgroups_parses_signed_high_gids_and_rejects_malformed() {
        assert_eq!(
            initgroups_line(b"alice -2147483648 -1 -1000 998\n", b"alice"),
            Some(vec![2_147_483_648, u32::MAX, 4_294_966_296, 998])
        );
        assert_eq!(initgroups_line(b"alice 2147483648 998\n", b"alice"), Some(vec![2_147_483_648, 998]));
        for bad in [&b"alice -0\n"[..], b"alice -\n", b"alice --1\n", b"alice -2147483649\n", b"alice +998\n"] {
            assert_eq!(initgroups_line(bad, b"alice"), None, "{bad:?}");
        }

        let dir = make_temp_dir("highgid");
        let prog = write_test_script(
            &dir,
            "fake-getent",
            "#!/bin/sh\n\
             [ \"$1\" = \"--\" ] || exit 9\n\
             shift\n\
             case \"$1:$2\" in\n\
               initgroups:alice) echo 'alice -2147483648 -1000 998'; exit 0 ;;\n\
               *) exit 2 ;;\n\
             esac\n",
        );
        assert!(initgroups_has_with(&prog, c"alice", 998).unwrap());
        assert!(initgroups_has_with(&prog, c"alice", 2_147_483_648).unwrap());
        assert!(!initgroups_has_with(&prog, c"alice", 1234).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
