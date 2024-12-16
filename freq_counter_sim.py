# Copyright (c) 2024 by Phase Advanced Sensor Systems, Inc.
# All rights reserved.
import argparse
import math
import time

import numpy as np
import glotlib

from psdb.util import prange


# What to display on the plot.
SHOW_REF_STEPS   = True
SHOW_RECIPROCAL  = False
SHOW_LINEAR_FIT  = False
SHOW_IIR_FITS    = True
LINE_WIDTH       = 2

# Display measurements using connected lines instead of steps.  Looks smoother
# but may obscure the data.  The reference frequency plot always uses steps.
RENDER_LINES     = True

# A list of IIR coefficients to plot.  The XHTIS-2 Modbus firmware uses a
# default coefficient of 9.
IIR_COEFS        = [9]

# The target crystal (pressure or temperature) prescaler which divides the
# incoming frequency pulses in order to make the number of interrupts that come
# in more manageable for the MCU.  The XHTIS-2 Modbus firmware uses a
# temperature crystal prescaler of 8 and a pressure crystal prescaler of 2.
PRESCALER        = 2

# The gate period in seconds to use for measuring the target crystal.  For the
# reciprocal and linear-fit measurements, this specifies the exact set of
# crystal pulses that are used to generate the measurement.  For the IIR filter
# measurement, this specifies the period with which to sample the IIR filter's
# output value.  A typical XtalX sensor uses an 0.1-second gate period.  The
# XHTIS-2 Modbus sensor, which uses IIR filtering exclusively, doesn't have a
# set gate period and instead returns the IIR filter's most recent output value
# whenever it is polled over Modbus; the host system therefore sets the period
# by choosing how often to perform a measurement over Modbus.
GATE_PERIOD_SECS = 0.1

# The number of gate periods for which to hold each frequency step.  Due to
# limitations in the simulation, frequency steps have to align with the end of a
# gate period.
GATES_PER_STEP   = 100

# A value that can be adjusted to simulate adding random jitter to the target
# crystal.  Very experimental.
JITTER           = 0 #0.0000000001


# Internal variables.
STEP_PERIOD = GATES_PER_STEP * GATE_PERIOD_SECS
K           = 1 #0.6781687 * 1.474560
RNG         = np.random.default_rng()


class Window(glotlib.Window):
    def __init__(self, x_counts, f0, f1, x_span, *args, **kwargs):
        super().__init__(*args, **kwargs)

        f_span         = (f1 - f0)
        self.x_counts  = x_counts
        self.pos_label = self.add_label((0.99, 0.01), '', anchor='SE')
        self.f_plot    = self.add_plot((3, 1, (1, 2)),
                                       limits=(0, f0 - .1*f_span,
                                               1.1*x_span, f1 + .1*f_span))

        counts_min  = min(np.min(counts) for counts in x_counts)
        counts_max  = max(np.max(counts) for counts in x_counts)
        ncounts     = max(len(counts) for counts in x_counts)
        c_span      = (counts_max - counts_min)
        self.c_plot = self.add_plot(313, limits=(-0.01 * ncounts,
                                                 -0.01 * c_span + counts_min,
                                                 1.01 * ncounts,
                                                 0.01 * c_span + counts_max))

        Y = x_counts[0]
        X = np.arange(len(Y))
        self.c_steps = self.c_plot.add_steps(X=X, Y=Y)
        self.displayed_x = 0

    def handle_mouse_moved(self, x, y):
        self.mark_dirty()

    def update_geometry(self, _t):
        updated = False

        _, _, p, data_x, data_y = self.get_mouse_pos()
        if data_x is not None:
            if p == self.f_plot:
                updated |= self.pos_label.set_text('%.10f  %.10f' %
                                                   (data_x, data_y))

                x = int(10 * data_x) + 1
                if x >= 0 and x < len(self.x_counts) and self.displayed_x != x:
                    Y = self.x_counts[x]
                    X = np.arange(len(Y))
                    self.c_steps.set_x_y_data(X, Y)
                    updated = True

        return updated


class GateGenerator:
    '''
    Class to help generate timing information for each pulse of a pressure
    crystal over a specified gate period in seconds.  This class can be called
    repeatedly to generate a new sequence of pressure pulse counts (the number
    of reference clock ticks that occurred during each pressure pulse).  The
    pressure pulse frequency is specified at the start of each gate period and
    is assumed to be a constant value for the full gate period.  This class
    currently cannot simulate changing the pressure frequency in the middle of
    a gate period.
    '''
    def __init__(self, ref_freq_hz, gate_period_secs, jitter):
        self.ref_freq_hz      = ref_freq_hz
        self.gate_period_secs = gate_period_secs
        self.gate_t1          = gate_period_secs
        self.jitter           = jitter
        self.pulse_count0     = 0
        self.pulse_t0         = 0

    def gen_gate_counts(self, pulse_freq):
        '''
        Generate one gate period of counts at the specified pulse_freq.
        '''
        pulse_period_secs = 1 / pulse_freq
        rem_t             = self.gate_t1 - self.pulse_t0
        N                 = int(rem_t // pulse_period_secs)
        jitter            = (self.jitter * (-1 + 2 * RNG.random(N))).cumsum()
        pulse_t1s         = (np.arange(1, N+1) * pulse_period_secs +
                             self.pulse_t0) + jitter
        while pulse_t1s[-1] >= self.gate_t1:
            pulse_t1s = pulse_t1s[:-1]
        count1s           = (pulse_t1s * self.ref_freq_hz).astype(int)
        counts            = np.diff(count1s, prepend=self.pulse_count0)
        self.pulse_count0 = count1s[-1]
        self.pulse_t0     = pulse_t1s[-1]
        self.gate_t1     += self.gate_period_secs # + 0.1 * RNG.random()

        return counts


class IIRFloat:
    '''
    This class uses double-precision math to implement a 5-stage IIR filter.
    Because the math is double-precision, it has enough resolution to not lose
    precision even when very small numbers are needed but it cannot be
    implemented on most of our microcontrollers because they only have a single-
    precision FPU which can't handle the necessary numerical resolution (the
    high-power Cortex-M7 found in the STM32H7 MCU does have a double-precision
    FPU and can implemented this version of the filter directly)..

    The floating-point version of the filter has the advantage that the filter
    coefficient can be any floating-point number, although we specify it here
    in the same form as the fixed-point code.
    '''
    def __init__(self, log2_coef, v=0):
        self.u    = [int(v)] * 5
        self.coef = 2**-log2_coef

    def reset(self, v=0):
        self.u = [int(v)] * 5

    def filt(self, v):
        self.u[0] += (v         - self.u[0]) * self.coef
        self.u[1] += (self.u[0] - self.u[1]) * self.coef
        self.u[2] += (self.u[1] - self.u[2]) * self.coef
        self.u[3] += (self.u[2] - self.u[3]) * self.coef
        self.u[4] += (self.u[3] - self.u[4]) * self.coef

    def to_freq_hz(self, ref_freq_hz):
        v = self.u[4]
        return ref_freq_hz * PRESCALER / (K * v)


class IIRFixed:
    '''
    This class uses fixed-point math to implement a 5-stage IIR filter.  The
    values are held in a 64-bit integer, with the low-order 46 bits being used
    as a fractional part.  The microcontrollers we use all embed 32-bit CPUs,
    however we can pretty easily emulate 64-bit integer math.

    The fixed-point version of the filter has the disadvantage that the
    filter coefficient has to be a power of two so that multiplication and
    division can be implemented as left- or right-shift operations.  Attempting
    to do true multiplication or division of 64-bit fixed-point numbers in
    real-time is beyond the capabilities of our MCUs.
    '''
    FILTER_RESOLUTION_BITS = 46

    def __init__(self, log2_coef, v=0):
        v = int(v)
        v *= K * 2**self.FILTER_RESOLUTION_BITS
        self.u  = [int(v)] * 5
        self.ia = log2_coef

    def reset(self, v=0):
        v *= K * 2**self.FILTER_RESOLUTION_BITS
        self.u = [int(v)] * 5

    def filt(self, v):
        v = (v << self.FILTER_RESOLUTION_BITS)
        self.u[0] += ((v         - self.u[0]) >> self.ia)
        self.u[1] += ((self.u[0] - self.u[1]) >> self.ia)
        self.u[2] += ((self.u[1] - self.u[2]) >> self.ia)
        self.u[3] += ((self.u[2] - self.u[3]) >> self.ia)
        self.u[4] += ((self.u[3] - self.u[4]) >> self.ia)

    def to_freq_hz(self, ref_freq_hz):
        period = (((((self.u[4] * 2**-self.FILTER_RESOLUTION_BITS) /
                     PRESCALER)) / ref_freq_hz) * K)
        return 1 / period


def gen_freq_start_stop(ref_freq_hz, counts):
    '''
    Given a list of reference clock counts for a sequence of pulses of the
    prescaled pressure crystal, estimate the frequency using reciprocal
    counting.
    '''
    Ey = counts.sum()
    N  = len(counts)
    return ref_freq_hz * N * PRESCALER / Ey


def gen_freq_fir(ref_freq_hz, counts):
    '''
    Given a list of reference clock counts for a sequence of pulses of the
    prescaled pressure crystal, estimate the frequency using the slope of the
    best linear fit to the data.
    '''
    N     = len(counts)
    X     = np.arange(N)
    Y     = np.cumsum(counts)
    Ey    = Y.sum()
    Exy   = (X*Y).sum()
    slope = 6*(2*Exy - (N-1)*Ey) / ((N-1)*(N+1)*N)
    return ref_freq_hz * PRESCALER / slope


def gen_freq_iir(ref_freq_hz, counts, iir):
    '''
    Given a list of reference clock counts for a sequence of pulses of the
    prescaled pressure crystal, use an IIR filter to estimate the frequency.
    Since an IIR filter is a continuous process that extends out to infinity,
    we also need to have the IIR filter's current state passed in as a
    parameter so that we can start with it and so it will be updated at the end
    of data processing.
    '''
    for c in counts:
        iir.filt(c)
    return iir.to_freq_hz(ref_freq_hz)


def main(args):
    ggj    = GateGenerator(args.ref_freq_hz, GATE_PERIOD_SECS, JITTER)
    f0     = args.pulse_freq_hz_0
    f1     = args.pulse_freq_hz_1
    period = args.modulation_period_secs
    nsteps = int(period // STEP_PERIOD)
    df     = (f1 - f0) / nsteps
    iirjs  = [IIRFixed(ia, v=(PRESCALER/f0)*args.ref_freq_hz)
              for ia in IIR_COEFS]

    ref_freqs   = []
    ss_freqs    = []
    fir_freqs   = []
    iirjs_freqs = [[] for _ in iirjs]
    x_counts    = []

    t_gen   = 0
    t_ss    = 0
    t_fir   = 0
    t_iirj  = 0
    t_start = time.time()
    sum_countsj = 0
    ncountsj = 0
    for i in prange(nsteps):
        if False:
            f = f0 if i < nsteps // 2 else f1
        else:
            f = f0 + df * i
        for _ in range(GATES_PER_STEP):
            t0 = time.time()
            countsj = ggj.gen_gate_counts(f / PRESCALER)
            sum_countsj += countsj.sum()
            ncountsj += len(countsj)
            x_counts.append(countsj)
            if SHOW_REF_STEPS:
                ref_freqs.append(f)
            t1 = time.time()
            if SHOW_RECIPROCAL:
                ss_freqs.append(gen_freq_start_stop(args.ref_freq_hz, countsj))
            t2 = time.time()
            if SHOW_LINEAR_FIT:
                fir_freqs.append(gen_freq_fir(args.ref_freq_hz, countsj))
            t3 = time.time()
            if SHOW_IIR_FITS:
                for iirj_freqs, iirj in zip(iirjs_freqs, iirjs):
                    iirj_freqs.append(gen_freq_iir(args.ref_freq_hz, countsj,
                                                   iirj))
            t4 = time.time()
            t_gen  += (t1 - t0)
            t_ss   += (t2 - t1)
            t_fir  += (t3 - t2)
            t_iirj += (t4 - t3)
    t_end = time.time()

    print('df %.10f f0 %.5f f1 %.5f' % (df, f0, f1))
    print('t_gen %s t_ss %s t_fir %s t_iirj %s t_total %s' %
          (t_gen, t_ss, t_fir, t_iirj, t_end - t_start))

    w = Window(x_counts, f0, f1, period, 2200, 1500, msaa=4)
    X0 = [0.1 * i for i in range(nsteps * GATES_PER_STEP)]
    if SHOW_REF_STEPS:
        print('Adding reference steps...')
        w.f_plot.add_steps(X=X0, Y=ref_freqs, width=LINE_WIDTH)
        w.f_plot.snap_bounds()
    if SHOW_RECIPROCAL:
        print('Adding reciprocal fit...')
        if RENDER_LINES:
            w.f_plot.add_lines(X=X0, Y=ss_freqs, width=LINE_WIDTH)
        else:
            w.f_plot.add_steps(X=X0, Y=ss_freqs, width=LINE_WIDTH)
    if SHOW_LINEAR_FIT:
        print('Adding linear fit...')
        if RENDER_LINES:
            w.f_plot.add_lines(X=X0, Y=fir_freqs, width=LINE_WIDTH)
        else:
            w.f_plot.add_steps(X=X0, Y=fir_freqs, width=LINE_WIDTH)
    if SHOW_IIR_FITS:
        for i, iirj_freqs in enumerate(iirjs_freqs):
            print('Adding IIR fit %u...' % i)
            if RENDER_LINES:
                w.f_plot.add_lines(X=X0, Y=iirj_freqs, width=LINE_WIDTH)
            else:
                w.f_plot.add_steps(X=X0, Y=iirj_freqs, width=LINE_WIDTH)
    print('Done.')

    glotlib.interact()


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-freq-hz', type=float, default=40e6)
    parser.add_argument('--pulse-freq-hz-0', type=float, default=50632.4)
    parser.add_argument('--pulse-freq-hz-1', type=float, default=50633.4)
    parser.add_argument('--modulation-period-secs', type=float, default=180)
    main(parser.parse_args())


if __name__ == '__main__':
    _main()
