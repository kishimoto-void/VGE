import numpy as np

# VGE4原型コアを内包
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
        self.pos += noise + 0.22 * total + 0.09 * memory
        self.pos = np.clip(self.pos, -3.8, 3.8)
        self.residue = 0.85 * self.residue + 0.07 * self.pos
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
    def __init__(self, n=2, seed=777, cooling_boost=1.25):
        self.layer_plus = VectorGenesisLayerPM(n=n, base_seed=seed, layer_sign=1)
        self.layer_minus = VectorGenesisLayerPM(n=n, base_seed=seed+500,
                                                layer_sign=-1, cooling_boost=cooling_boost)

    def step(self):
        plus_state = self.layer_plus.step()
        tension = plus_state["avg_dist"]
        assist = 0.0
        if tension > 0.55:
            assist = min((tension - 0.55) * 2.0, 0.95)
        elif tension > 0.3:
            assist = (tension - 0.3) * 1.0
        else:
            assist = max((0.3 - tension) * 0.35, 0)

        plus_center = self.layer_plus.get_vector()
        minus_inputs = []
        for v in self.layer_minus.voids:
            diff = v.pos - plus_center
            perp = np.array([-diff[1], diff[0]])
            fb = -0.18 * diff + 0.32 * perp * (1.0 + assist * 0.5)
            minus_inputs.append(fb)
        minus_state = self.layer_minus.step(external_inputs=minus_inputs, assist_from_other=assist)

        minus_center = self.layer_minus.get_vector()
        plus_feedback = []
        for v in self.layer_plus.voids:
            diff = v.pos - minus_center
            fb = -0.12 * diff
            plus_feedback.append(fb)
        for i, v in enumerate(self.layer_plus.voids):
            v.pos += 0.08 * plus_feedback[i]

        new_tension = abs(plus_state["avg_dist"] - minus_state["avg_dist"])
        return {
            "plus": plus_state,
            "minus": minus_state,
            "tension": round(new_tension, 4),
            "assist": round(assist, 3),
            "plus_dist": plus_state["avg_dist"],
            "minus_dist": minus_state["avg_dist"],
            "plus_center": plus_center,
            "minus_center": minus_center
        }

class Layer0:
    def __init__(self, attraction_strength=0.15):
        self.attraction_strength = attraction_strength

    def step(self, plus_center, minus_center):
        diff = plus_center - minus_center
        attraction_force = -self.attraction_strength * diff
        return attraction_force

    def apply_force(self, layer_plus, layer_minus, force):
        for v in layer_plus.voids:
            v.pos += force * 0.6
        for v in layer_minus.voids:
            v.pos += force * 0.8

class LightningRod:
    def __init__(self, guidance_strength=0.35, max_guidance=0.8):
        self.guidance_strength = guidance_strength
        self.max_guidance = max_guidance

    def guide_input(self, input_vector, equilibrium_point):
        if np.linalg.norm(input_vector) < 0.01:
            return input_vector, 0.0
        direction_to_eq = equilibrium_point - input_vector
        guidance = direction_to_eq * self.guidance_strength
        if np.linalg.norm(guidance) > self.max_guidance:
            guidance = guidance / np.linalg.norm(guidance) * self.max_guidance
        guided_vector = input_vector + guidance
        guidance_amount = np.linalg.norm(guidance)
        return guided_vector, guidance_amount

class VGE4_With_LightningRod:
    def __init__(self, seed=42):
        self.vge = VGE_LayerPlusMinus(n=2, seed=seed, cooling_boost=1.25)
        self.layer0 = Layer0(attraction_strength=0.15)
        self.lightning_rod = LightningRod(guidance_strength=0.35, max_guidance=0.8)
        self.emotional_bias = 0.0
        self.tension_maintenance_cost = 0.0
        self.lightning_rod_usage = 0.0
        self.history = []

    def set_emotional_bias(self, bias: float):
        self.emotional_bias = np.clip(bias, -1.0, 1.0)

    def step(self, over_input_strength=0.0, over_input_polarity=0.0):
        temp = self.vge.layer_plus.step()
        current_tension = temp["avg_dist"]
        plus_center = self.vge.layer_plus.get_vector()
        minus_center = self.vge.layer_minus.get_vector()
        equilibrium_point = (plus_center + minus_center) / 2

        input_density = 1.0
        rapid_cooling = False

        if current_tension > 0.65:
            input_density = 0.25
            rapid_cooling = True
            self.tension_maintenance_cost += 4.0
            for v in self.vge.layer_plus.voids + self.vge.layer_minus.voids:
                v.residue *= 0.45
        elif current_tension > 0.42:
            input_density = 0.5
            self.tension_maintenance_cost += 1.5

        if over_input_strength > 0:
            effective = over_input_strength * input_density
            raw_input = np.array([over_input_polarity * effective, over_input_polarity * effective * 0.5])
            guided_input, guidance_amount = self.lightning_rod.guide_input(raw_input, equilibrium_point)
            self.lightning_rod_usage += guidance_amount

            bias_eff = guided_input[0]
            for v in self.vge.layer_plus.voids:
                v.pos += np.array([bias_eff, bias_eff * 0.5]) * effective
            for v in self.vge.layer_minus.voids:
                v.pos += np.array([-bias_eff * 0.7, bias_eff * 0.3]) * effective

        state = self.vge.step()
        attraction = self.layer0.step(state["plus_center"], state["minus_center"])
        self.layer0.apply_force(self.vge.layer_plus, self.vge.layer_minus, attraction)

        plus_dist = state["plus_dist"]
        minus_dist = state["minus_dist"]
        tension = state["tension"]
        assist = state["assist"]

        emotion_balance = (plus_dist - minus_dist) / max(plus_dist + minus_dist, 0.01)
        emotion_balance = np.clip(emotion_balance + self.emotional_bias * 0.55, -1.0, 1.0)

        text = self._generate_text(emotion_balance, tension, assist, rapid_cooling, guidance_amount)

        record = {
            "tension": round(tension, 3),
            "assist": round(assist, 3),
            "emotion_balance": round(emotion_balance, 3),
            "lightning_rod_usage": round(self.lightning_rod_usage, 4),
            "maintenance_cost": round(self.tension_maintenance_cost, 2),
            "text": text
        }
        self.history.append(record)
        return record

    def _generate_text(self, balance, tension, assist, rapid_cooling, guidance_amount):
        if rapid_cooling:
            base = "急に冷やされて、避雷針が強く動いた"
        elif guidance_amount > 0.3:
            base = "入力が避雷針で均衡点に引導された"
        elif tension > 1.0:
            base = "熱がこみ上げてくるつつも、全体が纏がりを保ようとしている"
        elif tension > 0.6:
            base = "情緒が揺れながらも、均衡を保ようとする力が動いている"
        else:
            base = "穏やかで、全体が静かに均衡している"

        if balance > 0.35:
            mood = "明るく前向きに引力がかかっている"
        elif balance < -0.35:
            mood = "慎重で内省的に均衡が保たれている"
        else:
            mood = "相反する力がバランスよく動いている"

        return f"{base}。{mood}（テンション{tension:.1f} / 避雷針誤導{guidance_amount:.2f}）"

    def run_demo(self, steps=30):
        print("=== VGE4 + Layer 0 + 避雷針 プロトタイプ ===\n")
        print("入力ベクトルが避雷針で均衡点に引導されつつ、Layer 0が全体を纏がり止める\n")

        for t in range(steps):
            if t < 8:
                over_str, over_pol = 0.5, 0.7
            elif t < 15:
                over_str, over_pol = 1.6, 0.85
            elif t < 22:
                over_str, over_pol = 1.0, -0.5
            else:
                over_str, over_pol = 0.6, 0.3

            state = self.step(over_input_strength=over_str, over_input_polarity=over_pol)

            print(f"t={t:2d} | テンション={state['tension']:.2f} | アシスト={state['assist']:.2f} | "
                  f"避雷針使用={state.get('lightning_rod_usage', 0):.3f}")
            print(f"     → {state['text']}\n")

        print("=== デモ終了 ===")
        print(f"最終维持コスト: {self.tension_maintenance_cost:.1f}")
        print(f"避雷針総使用量: {self.lightning_rod_usage:.3f}")


if __name__ == "__main__":
    demo = VGE4_With_LightningRod(seed=777)
    demo.run_demo(28)