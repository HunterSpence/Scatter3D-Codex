# Security policy

## Supported versions

Security fixes are made on the default branch until the project begins publishing
stable releases. No tagged software release or historical release line is
currently supported; package metadata in the working tree is not a release
announcement.

## Reporting a vulnerability

Use GitHub's private **Report a vulnerability** form for this repository. Do not
open a public issue containing credentials, private measurement data, machine
addresses, or exploit details. Include the affected revision, a minimal
reproducer, impact, and any proposed mitigation. Expect an acknowledgement
within seven days; a repair timeline depends on severity and reproducibility.

## Data and execution safety

- Treat VNA exports, mesh files, and solver configuration as untrusted input.
- Run third-party meshes and plugins only in an isolated container or account.
- Never place instrument credentials or private datasets in manifests or issue
  attachments. Manifests contain hashes, not secrets.
- The CLI never controls a VNA or initiates RF emission. Instrument automation is
  intentionally out of scope for the initial release.
- CLI-generated reports and reconstruction artifacts are no-clobber by default.
  Treat `--force` as explicit authorization to replace an existing local
  artifact, not as an integrity or authenticity guarantee.
- The container is a reproducibility environment, not a hardened sandbox.
