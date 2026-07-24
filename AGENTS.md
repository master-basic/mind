# Batch scripting rules

- Inside `( ... )` blocks, escape parentheses in echo/text as `^( ^)` — an unescaped `)` silently truncates the block and leaks remaining lines into unconditional execution.
- Keep batch files ASCII-only. UTF-8 files render em dashes and other non-ASCII chars as mojibake (e.g. `ΓÇö`) on codepage 850/437 consoles. Use `chcp 65001 >nul` at the top if UTF-8 is required.
- `where` in PowerShell is an alias for `Where-Object`, not the cmd `where` command. For PATH searches inside batch files, use `where.exe` or `cmd /c where`.
- When locating an executable, don't rely solely on `where` / PATH. Fall back to well-known install paths and the registry (`HKLM\SOFTWARE\<Vendor>`).
- `fsutil fsinfo drives` requires admin on some systems — account for silent failure.
- Run scripts that need admin as: right-click → Run as administrator, or from an elevated cmd.exe.
- Always use full path for executables when calling them (e.g., `"!IMDISK!"` not bare `imdisk`).
- Use `setlocal EnableDelayedExpansion` and reference variables with `!var!` inside blocks.
