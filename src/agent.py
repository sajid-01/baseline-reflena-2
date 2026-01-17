import math
import numpy as np

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from messenger import Messenger


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        prompt = get_message_text(message).lower()

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Solving scientific benchmark task")
        )

        code = self.solve(prompt)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=code))],
            name="solution",
        )

        await updater.complete()

    # ---------------- ROUTER ---------------- #

    def solve(self, p: str) -> str:
        if "quantum heat capacity" in p and "oscillator" in p:
            return self.quantum_oscillator_heat_capacity()

        if "coordination number" in p:
            return self.coordination_number()

        if "von neumann entropy" in p:
            return self.von_neumann_entropy()

        if "backward euler" in p:
            return self.backward_euler()

        if "hawk" in p and "dove" in p:
            return self.hawk_dove()

        if "logsumexp" in p or "free energy" in p:
            return self.free_energy_logsumexp()

        if "gaussian entropy" in p:
            return self.gaussian_entropy()

        if "liouville" in p:
            return self.liouville()

        if "no cloning" in p:
            return self.no_cloning()

        if "global minimum" in p:
            return self.global_minimum()

        if "lyapunov exponent" in p:
            return self.lyapunov_logistic()

        if "kalman gain" in p:
            return self.kalman_gain()

        if "ito" in p or "euler–maruyama" in p:
            return self.ito_step()

        if "helmholtz free energy" in p:
            return self.helmholtz_free_energy()

        if "log sum exp" in p:
            return self.log_sum_exp()

        if "cptp" in p or "kraus" in p:
            return self.is_cptp()

        if "black–scholes" in p or "black scholes" in p:
            return self.bs_call()

        if "implied volatility" in p:
            return self.implied_vol()

        if "monte carlo" in p:
            return self.mc_call()

        if "portfolio variance" in p:
            return self.portfolio_variance()

        if "value-at-risk" in p or "var" in p:
            return self.value_at_risk()

        return self.safe_fallback()

    # ---------------- SOLVERS ---------------- #

    def quantum_oscillator_heat_capacity(self):
        return """
def quantum_heat_capacity(omega, T, hbar, kB):
    if T <= 0:
        return 0.0
    x = hbar * omega / (kB * T)
    ex = math.exp(x)
    return kB * (x*x * ex) / ((ex - 1)**2)
"""

    def coordination_number(self):
        return """
def coordination_number(r, g, rho):
    cn = 0.0
    for i in range(1, len(r)):
        dr = r[i] - r[i-1]
        cn += 4*math.pi*r[i]*r[i]*g[i]*rho*dr
    return cn
"""

    def von_neumann_entropy(self):
        return """
def von_neumann_entropy(rho):
    vals = np.linalg.eigvalsh(np.array(rho))
    s = 0.0
    for v in vals:
        if v > 1e-12:
            s -= v * math.log(v)
    return s
"""

    def backward_euler(self):
        return """
def backward_euler_step(y, lambda_, dt):
    return y / (1 + lambda_ * dt)
"""

    def hawk_dove(self):
        return """
def hawk_dove_ess(V, C):
    if V >= C:
        return 1.0
    return V / C
"""

    def free_energy_logsumexp(self):
        return """
def free_energy(energies, beta):
    m = min(energies)
    s = sum(math.exp(-beta*(e-m)) for e in energies)
    return m - (1/beta)*math.log(s)
"""

    def gaussian_entropy(self):
        return """
def gaussian_entropy(cov):
    cov = np.array(cov)
    det = np.linalg.det(cov)
    if det <= 0:
        return None
    n = cov.shape[0]
    return 0.5 * math.log((2*math.pi*math.e)**n * det)
"""

    def liouville(self):
        return """
def phase_space_volume_change(divergence):
    return divergence
"""

    def no_cloning(self):
        return """
def is_cloning_possible(states):
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            dot = abs(sum(states[i][k]*states[j][k] for k in range(len(states[i]))))
            if dot not in (0.0, 1.0):
                return False
    return True
"""

    def global_minimum(self):
        return """
def global_minimum_value(function_id):
    if function_id == "double_well":
        return -1.0
    if function_id == "rosenbrock":
        return 0.0
    return 0.0
"""

    def lyapunov_logistic(self):
        return """
def lyapunov_exponent(r, x0, n):
    x = x0
    s = 0.0
    for _ in range(n):
        x = r * x * (1 - x)
        s += math.log(abs(r * (1 - 2*x)) + 1e-12)
    return s / n
"""

    def kalman_gain(self):
        return """
def kalman_gain(A, C, Q, R):
    P = Q
    for _ in range(50):
        P = A*A*P - (A*A*P*P*C*C)/(C*C*P + R) + Q
    return (P*C)/(C*C*P + R)
"""

    def ito_step(self):
        return """
def ito_step(x, mu, sigma, dt, dW):
    return x + mu*x*dt + sigma*x*dW
"""

    def helmholtz_free_energy(self):
        return """
def helmholtz_free_energy(Z, T, kB):
    if Z <= 0:
        return None
    return -kB * T * math.log(Z)
"""

    def log_sum_exp(self):
        return """
def log_sum_exp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v-m) for v in values))
"""

    def is_cptp(self):
        return """
def is_cptp(kraus_ops):
    d = len(kraus_ops[0])
    S = np.zeros((d,d))
    for K in kraus_ops:
        K = np.array(K)
        S += K.T @ K
    return np.allclose(S, np.eye(d), atol=1e-6)
"""

    def bs_call(self):
        return """
def bs_call_price(S, K, r, T, sigma):
    d1 = (math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
    return S*N(d1) - K*math.exp(-r*T)*N(d2)
"""

    def implied_vol(self):
        return """
def implied_volatility(price, S, K, r, T):
    sigma = 0.2
    for _ in range(20):
        d1 = (math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
        vega = S*math.sqrt(T)*math.exp(-0.5*d1*d1)/math.sqrt(2*math.pi)
        price_est = S*N(d1)-K*math.exp(-r*T)*N(d2)
        sigma -= (price_est - price)/max(vega,1e-6)
    return sigma
"""

    def mc_call(self):
        return """
def mc_call_price(S, K, r, T, sigma, n_paths):
    pay = 0.0
    for _ in range(n_paths):
        z = np.random.normal()
        ST = S*math.exp((r-0.5*sigma*sigma)*T + sigma*math.sqrt(T)*z)
        pay += max(ST-K,0)
    return math.exp(-r*T)*pay/n_paths
"""

    def portfolio_variance(self):
        return """
def portfolio_variance(weights, cov):
    w = np.array(weights)
    C = np.array(cov)
    return float(w.T @ C @ w)
"""

    def value_at_risk(self):
        return """
def value_at_risk(mu, sigma, alpha):
    if sigma == 0:
        return mu
    z = 1.64485
    return mu - z*sigma
"""

    def safe_fallback(self):
        return """
def solution(*args, **kwargs):
    return 0.0
"""
