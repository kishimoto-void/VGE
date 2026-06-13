import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class VectorGenesisVoid2D:
    def __init__(self, seed=None, name="", pos=None, coupling_sign=-1.0, rotation_strength=0.0):
        self.rng = np.random.RandomState(seed)
        self.name = name
        self.pos = np.array(pos if pos is not None else [0.0, 0.0], dtype=float)
        self.coupling_sign = coupling_sign
        self.rotation_strength = rotation_strength
        self.residue = np.zeros(2)

    def step(self, others, external_input=np.zeros(2)):
        noise = self.rng.uniform(-0.1, 0.1, 2)
        if others:
            mean_pos = np.mean([o.pos for o in others], axis=0)
            diff = self.pos - mean_pos
            force = self.coupling_sign * diff
            if abs(self.rotation_strength) > 0.01:
                perp = np.array([-diff[1], diff[0]])
                force += self.rotation_strength * perp
        else:
            force = np.zeros(2)
        total = force + external_input
        memory = np.tanh(self.residue)
        self.pos += noise + 0.10 * total + 0.06 * memory
        self.pos = np.clip(self.pos, -3.8, 3.8)
        self.residue = 0.87 * self.residue + 0.05 * self.pos
        return {"pos": self.pos.copy(), "force": force.copy()}

class VectorGenesisLayerPM:
    def __init__(self, n=2, base_seed=42, layer_sign=1, cooling_boost=1.0):
        self.n = n
        self.layer_sign = layer_sign
        self.cooling_boost = cooling_boost
        self.voids = []
        for i in range(n):
            seed = base_seed + i * 7
            init = np.array([np.random.RandomState(seed).uniform(-1.0, 1.0),
                             np.random.RandomState(seed+3).uniform(-1.0, 1.0)])
            if layer_sign == 1:
                rot = 0.12
                coup = -1.0
            else:
                rot = -0.42 * cooling_boost
                coup = -0.55
            v = VectorGenesisVoid2D(seed=seed, name=f"L{layer_sign}{i+1}",
                                    pos=init, coupling_sign=coup, rotation_strength=rot)
            self.voids.append(v)

    def step(self, external_inputs=None, assist_from_other=0.0):
        if external_inputs is None:
            external_inputs = [np.zeros(2)] * self.n
        states = []
        for i, v in enumerate(self.voids):
            others = [self.voids[j] for j in range(self.n) if j != i]
            ext = external_inputs[i] + assist_from_other * 0.08
            st = v.step(others, ext)
            states.append(st)
        positions = np.array([s["pos"] for s in states])
        center = np.mean(positions, axis=0)
        avg_dist = np.mean(np.linalg.norm(positions - center, axis=1))
        return {"positions": positions, "center": center, "avg_dist": round(avg_dist, 4)}

    def get_vector(self):
        pos = np.array([v.pos for v in self.voids])
        return np.mean(pos, axis=0)

class VGE_LayerPlusMinus:
    """
    Layer +1 × Layer -1 ：非対称モデル（自己発散 + 逆回転冷却・アシスト）
    プロトコア・非対称 4連装モデル
    """
    def __init__(self, n=2, seed=777, cooling_boost=1.25):
        self.layer_plus = VectorGenesisLayerPM(n=n, base_seed=seed, layer_sign=1)
        self.layer_minus = VectorGenesisLayerPM(n=n, base_seed=seed+500,
                                                layer_sign=-1, cooling_boost=cooling_boost)

    def step(self):
        plus_state = self.layer_plus.step()
        tension = plus_state["avg_dist"]
        assist = 0.0
        if tension > 0.8:
            assist = min((tension - 0.8) * 1.8, 0.9)
        elif tension > 0.4:
            assist = (tension - 0.4) * 0.6
        plus_center = self.layer_plus.get_vector()
        minus_inputs = []
        for v in self.layer_minus.voids:
            diff = v.pos - plus_center
            perp = np.array([-diff[1], diff[0]])
            fb = -0.15 * diff + 0.28 * perp * (1.0 + assist * 0.6)
            minus_inputs.append(fb)
        minus_state = self.layer_minus.step(external_inputs=minus_inputs, assist_from_other=assist)
        minus_center = self.layer_minus.get_vector()
        plus_feedback = []
        for v in self.layer_plus.voids:
            diff = v.pos - minus_center
            fb = -0.05 * diff
            plus_feedback.append(fb)
        for i, v in enumerate(self.layer_plus.voids):
            v.pos += 0.03 * plus_feedback[i]
        new_tension = abs(plus_state["avg_dist"] - minus_state["avg_dist"])
        return {
            "plus": plus_state,
            "minus": minus_state,
            "tension": round(new_tension, 4),
            "assist": round(assist, 3),
            "plus_dist": plus_state["avg_dist"],
            "minus_dist": minus_state["avg_dist"]
        }

    def run(self, steps=200):
        history = []
        for _ in range(steps):
            history.append(self.step())
        return history


if __name__ == "__main__":
    print("=== Layer +1 × Layer -1 非対称 4連装プロトコアモデル ===")
    system = VGE_LayerPlusMinus(n=2, seed=777, cooling_boost=1.25)
    hist = system.run(150)
    last = hist[-1]
    print(f"Final Tension: {last['tension']}")
    print(f"+1 dist: {last['plus_dist']:.4f}  -1 dist: {last['minus_dist']:.4f}")
    print(f"Assist level: {last['assist']}")