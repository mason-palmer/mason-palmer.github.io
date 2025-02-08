---
title: "Understanding Timing Error Characteristics from Overclocked Systolic Multiply–Accumulate Arrays in FPGAs"
collection: publications
category: manuscripts
permalink: /publication/Understanding-Timing-Error-Characteristics-from-Overclocked-Systolic-Multiply–Accumulate-Arrays-in-FPGAs
excerpt: 'A study on timing errors in AI hardware accelerators in overclocked or under-volted conditions.'
date: 2024-01-09
venue: 'Journal of Low Power Electronics and Applications'
paperurl: 'http://mason-palmer.github.io/files/UnderstandingTimingError.pdf'
citation: 'Chamberlin, Andrew, Andrew Gerber, Mason Palmer, Tim Goodale, Noel Daniel Gundi, Koushik Chakraborty, and Sanghamitra Roy. 2024. "Understanding Timing Error Characteristics from Overclocked Systolic Multiply–Accumulate Arrays in FPGAs" Journal of Low Power Electronics and Applications 14, no. 1: 4. https://doi.org/10.3390/jlpea14010004'
---

**Abstract**: Artificial Intelligence (AI) hardware accelerators have seen tremendous developments in
recent years due to the rapid growth of AI in multiple fields. Many such accelerators comprise a
Systolic Multiply–Accumulate Array (SMA) as its computational brain. In this paper, we investigate
the faulty output characterization of an SMA in a real silicon FPGA board. Experiments were run on
a single Zybo Z7-20 board to control for process variation at nominal voltage and in small batches to
control for temperature. The FPGA is rated up to 800 MHz in the data sheet due to the max frequency
of the PLL, but the design is written using Verilog for the FPGA and C++ for the processor and
synthesized with a chosen constraint of a 125 MHz clock. We then operate the system at a frequency
range of 125 MHz to 450 MHz for the FPGA and the nominal 667 MHz for the processor core to
produce timing errors in the FPGA without affecting the processor. Our extensive experimental
platform with a hardware–software ecosystem provides a methodological pathway that reveals
fascinating characteristics of SMA behavior under an overclocked environment. While one may
intuitively expect that timing errors resulting from overclocked hardware may produce a wide
variation in output values, our post-silicon evaluation reveals a lack of variation in erroneous output
values. We found an intriguing pattern where error output values are stable for a given input across
a range of operating frequencies far exceeding the rated frequency of the FPGA.
