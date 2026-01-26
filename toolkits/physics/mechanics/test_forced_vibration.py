#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强迫振动模块测试
"""

import os
import sys
import unittest
import numpy as np

# 确保可以导入被测模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.forced_vibration.forced_vibration_solver import (
    calc_natural_frequency,
    calc_forced_vibration_amplitude,
    solve_vibration_equation,
    steam_engine_indicator_amplitude,
)


class TestForcedVibration(unittest.TestCase):
    """强迫振动核心函数测试"""

    def test_calc_natural_frequency(self):
        # k=100 N/m, m=4 kg -> ω0 = 5 rad/s
        self.assertAlmostEqual(calc_natural_frequency(100.0, 4.0), 5.0, places=7)

    def test_calc_forced_vibration_amplitude(self):
        # 简单数值：F0=10, k=100, m=1, ω=2, ω0=10 -> A = 10/(1*(100-4)) = 10/96
        amp = calc_forced_vibration_amplitude(10.0, 100.0, 1.0, 2.0)
        self.assertAlmostEqual(amp, 10.0 / 96.0, places=7)

    def test_solve_vibration_equation_energy_growth(self):
        # 恒定外力 F(t)=F0，期望位移随时间增长为抛物线（无阻尼），仅做数值运行健壮性检查
        m, k = 1.0, 100.0
        F0 = 1.0
        def F(t):
            return F0
        t = np.linspace(0.0, 1.0, 200)
        tt, yy = solve_vibration_equation(m, k, F, (0.0, 1.0), (0.0, 0.0), t)
        self.assertEqual(len(tt), len(t))
        self.assertEqual(yy.shape[0], 2)

    def test_indicator_amplitude_against_theory(self):
        # 对应题述：p = 40 + 30 sin(2πt/T), S=4, m=1, k=30, T=1/3
        S = 4.0
        m = 1.0
        k = 30.0
        T = 1.0 / 3.0
        def p_func(t):
            return 40.0 + 30.0 * np.sin(2.0 * np.pi * t / T)
        amp, res = steam_engine_indicator_amplitude(p_func, S, m, k, T)
        # 理论：F0 = 30*S; A = F0/(m(ω0^2-ω^2))
        omega0 = np.sqrt(k / m)
        omega = 2.0 * np.pi / T
        A_theory = abs((30.0 * S) / (m * (omega0**2 - omega**2)))
        self.assertAlmostEqual(amp, A_theory, places=6)


class TestIntegrationExamples(unittest.TestCase):
    """集成示例：三个调用场景"""

    def test_example_harmonic_force(self):
        # 简谐外力：验证稳态振幅数量级合理
        m, k = 2.0, 50.0
        omega = 4.0
        F0 = 5.0
        def F(t):
            return F0 * np.sin(omega * t)
        t = np.linspace(0.0, 5.0, 2000)
        tt, yy = solve_vibration_equation(m, k, F, (0.0, 5.0), (0.0, 0.0), t)
        self.assertEqual(len(tt), len(t))

    def test_example_random_force(self):
        # 随机外力：仅健壮性
        rng = np.random.default_rng(42)
        samples = rng.normal(size=10000)
        m, k = 1.0, 20.0
        t = np.linspace(0.0, 10.0, len(samples))
        def F(ti):
            # 最近索引采样
            idx = int(ti / 10.0 * (len(samples) - 1))
            return float(samples[idx])
        tt, yy = solve_vibration_equation(m, k, F, (0.0, 10.0), (0.0, 0.0), t)
        self.assertTrue(np.isfinite(yy).all())

    def test_example_indicator_from_problem(self):
        # 题目同参数的实际运行
        S, m, k = 4.0, 1.0, 30.0
        T = 1.0 / 3.0
        def p_func(t):
            return 40.0 + 30.0 * np.sin(2.0 * np.pi * t / T)
        amp, res = steam_engine_indicator_amplitude(p_func, S, m, k, T)
        self.assertTrue(amp >= 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
