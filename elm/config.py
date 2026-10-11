"""Parse an elm-olmt ``.cfg`` into the metadata the engine needs.

The ``.cfg`` is INI-like but hand-edited: it uses ``#`` for comments (including
inline), values may be single-quoted, and booleans appear in Fortran style
(``.true.`` / ``.false.``). This parser is deliberately tolerant and extracts
only the fields the EFC engine requires:

    [simulation]  sites, case_names
    [run_lengths] trans_startyear, nyears_trans
    [case_options] surffile, use_vertsoilc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseConfig:
    site: str
    case_name: str
    surffile: Path
    use_vertsoilc: bool
    trans_startyear: int
    nyears_trans: int

    @property
    def end_year(self) -> int:
        """Inclusive final transient year."""
        return self.trans_startyear + self.nyears_trans - 1


def _strip_inline_comment(value: str) -> str:
    # Remove an unquoted trailing '#...' comment.
    out, quote = [], None
    for ch in value:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        return value[1:-1].strip()
    return value


def _parse_bool_fortran(value: str) -> bool:
    v = _unquote(value).strip().lower().strip(".")
    if v in ("true", "t"):
        return True
    if v in ("false", "f"):
        return False
    raise ValueError(f"Cannot parse Fortran boolean from {value!r}")


def parse_cfg(path: str | Path) -> dict[str, dict[str, str]]:
    """Return a {section: {key: raw_value}} mapping (comments stripped)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if "=" not in line or current is None:
            continue
        key, _, value = line.partition("=")
        sections[current][key.strip()] = _strip_inline_comment(value)
    return sections


def load_case_config(path: str | Path) -> CaseConfig:
    """Parse a ``.cfg`` into a validated :class:`CaseConfig`."""
    sections = parse_cfg(path)

    def need(section: str, key: str) -> str:
        if section not in sections or key not in sections[section]:
            raise KeyError(f"{Path(path).name}: missing [{section}] {key}")
        return sections[section][key]

    site = _unquote(need("simulation", "sites"))
    case_name = _unquote(need("simulation", "case_names"))
    surffile = Path(_unquote(need("case_options", "surffile")))
    use_vertsoilc = _parse_bool_fortran(need("case_options", "use_vertsoilc"))
    trans_startyear = int(_unquote(need("run_lengths", "trans_startyear")))
    nyears_trans = int(_unquote(need("run_lengths", "nyears_trans")))

    if "," in site or "," in case_name:
        raise ValueError(
            f"{Path(path).name}: multiple sites/cases not supported "
            f"(got sites={site!r}, case_names={case_name!r})"
        )
    return CaseConfig(
        site=site,
        case_name=case_name,
        surffile=surffile,
        use_vertsoilc=use_vertsoilc,
        trans_startyear=trans_startyear,
        nyears_trans=nyears_trans,
    )
