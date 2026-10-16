# Bill of Materials (as built)

Values are what was actually assembled, including local substitutions. Signal-path capacitors are film type for stability. The reasoning behind each substitution is in [`../docs/design-decisions.md`](../docs/design-decisions.md).

| Block | Component | Value / Part | Note |
|---|---|---|---|
| Reference buffer | Op-amp | OPA333AIDR | Precision, low input offset voltage |
| Reference divider | Resistors ×2 | 100 kΩ | Sets 1.65 V from a 3.3 V supply |
| Signal op-amp | Dual op-amp | MCP602-I/P | Charge amp and gain stage on one chip |
| Charge-amp feedback | Resistor (Rf) | 1 MΩ | Changed from the specified 100 MΩ (availability) |
| Charge-amp feedback | Capacitor (Cf) | 940 nF (2× 470 nF film, parallel) | Raised alongside the Rf change to keep the same filter corner |
| Gain stage | Resistors | 68 kΩ + 1 kΩ | Gain = 1 + 68k/1k = ×69. The 1 kΩ leg returns to the reference rail, not ground |
| Low-pass filter | Resistor + capacitor | 10 kΩ + film cap | Anti-aliasing ahead of the ADC |
| ADC protection | Resistor + capacitor | 1 kΩ + 10 nF film | Current limit and filtering at the ADC input pin |
| Input clamp | Dual diode | BAV199LT1G | Low reverse leakage, protects the high-impedance input |
| Local decoupling | Capacitors | 100 nF ×2 | One per IC |
| Bulk decoupling | Electrolytic | 10 µF | At the supply entry only. Never on the reference rail (see the debugging log) |

## Component selection method

Every substitution above followed the same method:

1. Identify the part's job in this circuit.
2. Identify the one spec that would break the design if wrong.
3. Match that spec tightly and treat everything else as good enough.

The full write-up, including a rejected part, is in [`../docs/design-decisions.md`](../docs/design-decisions.md).
