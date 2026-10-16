# Design Decisions

## Charge amplifier over voltage mode

The PVDF film is a capacitor that generates charge when flexed. The charge per unit of force is reliable, but the voltage the film develops on its own depends on its capacitance, which drifts with temperature and varies between units. This was confirmed on the bench: pressing the film changes the measured voltage but not its capacitance. Voltage is just charge divided by a capacitance that isn't fixed.

For anything built in volume, reading voltage directly means inconsistent readings across units.

A charge amplifier holds the sensor at a fixed reference voltage, so it never develops a voltage of its own. The generated charge is forced onto a precision feedback capacitor (C<sub>f</sub>) instead:

```
Vout = Q / Cf
```

The sensor's own capacitance drops out of the equation, so every unit behaves the same regardless of sensor tolerance or temperature. This is the main reason for the architecture, and it's the standard front end for piezoelectric sensors.

### The rejected alternative

A bleed resistor plus instrumentation amp (voltage mode) was evaluated first. With a ~2 nF sensor, passing a ~1 Hz heartbeat cleanly needs a bleed resistor around 150 MΩ, and the instrumentation amp's input bias current through that resistor produces enough offset to saturate the gain stage. Its high-pass corner also depends on the sensor's own capacitance, which brings back the unit-to-unit inconsistency the charge amp avoids. It was kept as a documented fallback.

## Component selection: the dominant-spec method

Locally available parts rarely match an exact datasheet spec, so every substitution followed the same method:

1. Identify the part's job in this circuit.
2. Identify the one failure mode that would break the design.
3. Match that spec tightly. Everything else only needs to be good enough.

| Part | Job | Spec that matters | Why |
|---|---|---|---|
| MCP602 | Both signal op-amps | Input bias current | The very high input impedance turns a small current into a large error, so it needs a CMOS input |
| OPA333 | Buffer the VREF rail | Input offset voltage | Any offset lands directly on the reference the whole circuit is built around |
| BAV199 | Input protection clamp | Reverse leakage | Leakage drains charge off the input node and corrupts the signal |

An LM324 was ruled out early for this reason: its bipolar input bias current, multiplied by this circuit's high resistance, would have produced an offset large enough to break the design.

## Why the VREF buffer is needed

The circuit runs on a single 3.3 V supply, so the signal is centred on a 1.65 V reference (VREF) to let it swing both ways. A bare resistor divider sags whenever something draws current from it, which injects noise into a signal measured in microvolts. The OPA333 buffer presents VREF as a stiff, low-impedance source.

A bulk capacitor on VREF cannot be used instead. The sensor's second terminal also connects to VREF, so a bulk cap there gives the sensor's charge a low-impedance path to ground instead of onto the feedback capacitor, and the signal disappears. This was found by experiment; see [`debugging-log.md`](debugging-log.md).

## Feedback resistor: 100 MΩ to 1 MΩ

The specified 100 MΩ feedback resistor wasn't available locally. The filter corner depends on the product of the resistor and capacitor:

```
f_HP = 1 / (2π · Rf · Cf)
```

So Rf was reduced 100× to 1 MΩ and Cf raised to 940 nF to keep the same corner. Rf's only independent effect is on DC error, which scales with resistance, so the smaller value is slightly better. The op-amp's near-zero input current never needed a resistor as large as 100 MΩ.

## Signal chain

```
PVDF sensor -> Stage 1: charge amp (charge to voltage, HPF ~0.17 Hz)
            -> Stage 2: gain ×69 (non-inverting, referenced to VREF)
            -> Stage 3: low-pass filter (anti-aliasing)
            -> Stage 4: ADC protection (current limit and final filtering)
            -> ADC
```

Both the heartbeat and the much slower breathing signal are extracted from this one waveform in software, so no second analog channel is needed.
