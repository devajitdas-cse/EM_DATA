# EM_DATA
This repository contains the electromagnetic leakage dataset and analysis scripts used for LNA-free SDR-based measurement of workload-dependent activity in an edge IoT device.

The dataset was collected from a Raspberry Pi 4B using an RTL-SDR receiver and an H-field near-field probe without any external low-noise amplifier. Complex baseband I/Q traces were recorded under controlled Idle and CPU Load states. The recordings were used to evaluate adaptive electromagnetic leakage band selection across multiple center frequencies.

## Dataset Overview

The repository includes EM leakage recordings collected at the following center frequencies:

* 80 MHz
* 100 MHz
* 120 MHz
* 150 MHz
* 200 MHz

The dataset contains short scouting traces and longer validation traces. The scouting traces are used for frequency-band ranking, while the validation traces are used to compare the selected band with a fixed-frequency baseline.

## Acquisition Setup

* Device under test: Raspberry Pi 4B
* Receiver: RTL-SDR Blog V3
* Probe: H-field near-field magnetic probe
* Acquisition type: LNA-free SDR measurement
* File format: Complex64 `.cfile`
* Operating states: Idle and CPU Load
* Load generation: `stress-ng --cpu 4`

## Repository Contents

```text
EM_DATA/
├── idle/                         # Idle-state EM traces
├── load/                         # CPU Load-state EM traces
├── validation/                   # Long-duration validation traces
├── trace_index.csv               # Metadata for scouting traces
├── validation_trace_index.csv    # Metadata for validation traces
├── validate_lna_free_bands.py    # Main validation script
└── validate_any_index.py         # General trace-index validation script
```

## Purpose

This dataset supports research on low-cost electromagnetic leakage measurement, LNA-free SDR acquisition, adaptive frequency-band selection, and workload-dependent EM side-channel analysis for edge IoT devices.




## License

Please check the repository license before using or redistributing the dataset.

