"""Deterministic netlist parser for CPT v2.8 Circuit Oracle Core.

Supports a minimal one-component-per-line format:
  V<name> <positive> <negative> <voltage>
  R<name> <node_a> <node_b> <resistance>
  I<name> <positive> <negative> <current>

Empty lines and # comments are ignored. Node '0' or 'GND' normalized to '0'.
"""

from __future__ import annotations

from backend.circuits.models import Circuit, CurrentSource, Resistor, VoltageSource, _normalize_ground


def parse_netlist(text: str, name: str = "unnamed") -> Circuit:
    """Parse a netlist string into a Circuit. Deterministic: components sorted by name."""
    resistors: list[Resistor] = []
    voltage_sources: list[VoltageSource] = []
    current_sources: list[CurrentSource] = []

    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(f"Line {line_num}: expected at least 4 tokens, got {len(tokens)}: {raw_line!r}")

        designator = tokens[0]
        # First char determines type
        prefix = designator[0].upper()

        try:
            if prefix == "R":
                if len(tokens) != 4:
                    raise ValueError(f"Line {line_num}: resistor requires 4 tokens (R<name> <node_a> <node_b> <resistance>), got {len(tokens)}")
                r_name = designator
                node_a = _normalize_ground(tokens[1])
                node_b = _normalize_ground(tokens[2])
                resistance = float(tokens[3])
                if resistance <= 0:
                    raise ValueError(f"Line {line_num}: resistance must be > 0, got {resistance}")
                resistors.append(Resistor(name=r_name, node_a=node_a, node_b=node_b, resistance_ohm=resistance))

            elif prefix == "V":
                if len(tokens) != 4:
                    raise ValueError(f"Line {line_num}: voltage source requires 4 tokens (V<name> <pos> <neg> <voltage>), got {len(tokens)}")
                v_name = designator
                pos = _normalize_ground(tokens[1])
                neg = _normalize_ground(tokens[2])
                voltage = float(tokens[3])
                voltage_sources.append(VoltageSource(name=v_name, positive=pos, negative=neg, voltage=voltage))

            elif prefix == "I":
                if len(tokens) != 4:
                    raise ValueError(f"Line {line_num}: current source requires 4 tokens (I<name> <pos> <neg> <current>), got {len(tokens)}")
                i_name = designator
                pos = _normalize_ground(tokens[1])
                neg = _normalize_ground(tokens[2])
                current = float(tokens[3])
                if current == 0:
                    raise ValueError(f"Line {line_num}: current source must be non-zero, got 0")
                current_sources.append(CurrentSource(name=i_name, positive=pos, negative=neg, current=current))

            else:
                raise ValueError(f"Line {line_num}: unknown component prefix '{prefix}' in '{designator}'")

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Line {line_num}: malformed line: {raw_line!r} — {exc}") from exc

    if not resistors and not voltage_sources and not current_sources:
        raise ValueError("Netlist contains no components")

    # Circuit.__post_init__ sorts by name
    return Circuit(
        name=name,
        resistors=tuple(resistors),
        voltage_sources=tuple(voltage_sources),
        current_sources=tuple(current_sources),
    )
