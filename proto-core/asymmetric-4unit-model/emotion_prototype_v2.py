import numpy as np

# VGE4原型コアを内包（簡易版）
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
        # 改善: 外部入力の影響を強化
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
        # 改善: アシストの感度を上げる
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
            # 改善: Layer間フィードバックを強化
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
            v.pos += 0.08 * plus_feedback[i]  # 強化

        new_tension = abs(plus_state["avg_dist"] - minus_state["avg_dist"])
        return {
            "plus": plus_state,
            "minus": minus_state,
            "tension": round(new_tension, 4),
            "assist": round(assist, 3),
            "plus_dist": plus_state["avg_dist"],
            "minus_dist": minus_state["avg_dist"]
        }


class SafeEmotionVGE:
    def __init__(self, seed=42):
        self.vge = VGE_LayerPlusMinus(n=2, seed=seed, cooling_boost=1.25)
        self.emotional_bias = 0.0
        self.desired_tension = 0.5
        self.tension_maintenance_cost = 0.0
        self.history = []

    def set_emotional_bias(self, bias: float):
        self.emotional_bias = np.clip(bias, -1.0, 1.0)

    def set_desired_tension(self, tension: float):
        self.desired_tension = np.clip(tension, 0.0, 2.0)

    def inject_over_input(self, strength: float = 1.0, polarity: float = 0.0, input_density: float = 1.0):
        effective_strength = strength * input_density
        bias_effect = polarity * effective_strength * 0.45
        for v in self.vge.layer_plus.voids:
            v.pos += np.array([bias_effect, bias_effect * 0.5]) * effective_strength
        for v in self.vge.layer_minus.voids:
            v.pos += np.array([-bias_effect * 0.7, bias_effect * 0.3]) * effective_strength

    def step(self, over_input_strength=0.0, over_input_polarity=0.0):
        # 現在のテンションを取得
        temp_state = self.vge.layer_plus.step()
        current_tension = temp_state["avg_dist"]

        input_density = 1.0
        rapid_cooling_applied = False

        # === 安全設計 ===
        if current_tension > 0.65:          # 改善: 実際的な値に下げる
            # 急速冷却（冷水ぶっかける）
            input_density = 0.25
            rapid_cooling_applied = True
            self.tension_maintenance_cost += 4.0

            # 実際に冷却する（residueを減衰）
            for v in self.vge.layer_plus.voids:
                v.residue *= 0.45
            for v in self.vge.layer_minus.voids:
                v.residue *= 0.45

        elif current_tension > 0.42:      # 改善: 予防カットの閾値を下げる
            input_density = 0.5
            self.tension_maintenance_cost += 1.5

        if over_input_strength > 0:
            self.inject_over_input(over_input_strength, over_input_polarity, input_density)

        state = self.vge.step()

        plus_dist = state["plus_dist"]
        minus_dist = state["minus_dist"]
        tension = state["tension"]
        assist = state["assist"]

        emotion_balance = (plus_dist - minus_dist) / max(plus_dist + minus_dist, 0.01)
        emotion_balance = np.clip(emotion_balance + self.emotional_bias * 0.55, -1.0, 1.0)

        text = self._generate_emotional_text(emotion_balance, tension, assist, rapid_cooling_applied)

        record = {
            "tension": round(tension, 3),
            "assist": round(assist, 3),
            "emotion_balance": round(emotion_balance, 3),
            "plus_dist": round(plus_dist, 3),
            "minus_dist": round(minus_dist, 3),
            "input_density": round(input_density, 2),
            "rapid_cooling": rapid_cooling_applied,
            "maintenance_cost": round(self.tension_maintenance_cost, 2),
            "text": text
        }
        self.history.append(record)
        return record

    def _generate_emotional_text(self, balance, tension, assist, rapid_cooling):
        if rapid_cooling:
            base = "急に冷やされて、情緒が少し引き締められた感じ"
        elif tension > 1.0:
            if assist > 0.25:
                base = "熱がこみ上げてくるけど、どこかで冷静に整えられているような"
            else:
                base = "激しく情緒が揺さぶられて、言葉が溢れそうになる"
        elif tension > 0.6:
            base = "情緒が少し揺れて、複雑な気持ちが混ざっている"
        else:
            base = "穏やかで、情緒が静かに流れている"

        if balance > 0.35:
            mood = "明るく前向きで、希望や喜びが感じられる"
        elif balance > 0.05:
            mood = "穏やかながらも少し期待が膨らんでいる"
        elif balance < -0.35:
            mood = "少し重く、慎重で内省的な響きがある"
        else:
            mood = "中間的で、さまざまな情緒が混ざり合っている"

        return f"{base}。{mood}（テンション{tension:.1f} / アシスト{assist:.1f}）"

    def run_demo(self, steps=30):
        print("=== VGE4 情緒プロトタイプ v2（安全設計版） ===\n")
        print("・テンション > 0.42 → 入力50%カット（予防）")
        print("・テンション > 0.65 → 急速冷却 + コスト加算\n")

        for t in range(steps):
            if t < 8:
                self.set_desired_tension(0.4)
                over_str, over_pol = 0.5, 0.7
            elif t < 15:
                self.set_desired_tension(1.4)
                over_str, over_pol = 1.6, 0.85
            elif t < 22:
                self.set_desired_tension(1.0)
                over_str, over_pol = 1.0, -0.5
            else:
                self.set_desired_tension(0.65)
                over_str, over_pol = 0.6, 0.3

            state = self.step(over_input_strength=over_str, over_input_polarity=over_pol)

            cooling_mark = "【急速冷却】" if state["rapid_cooling"] else ""
            print(f"t={t:2d} | テンション={state['tension']:.2f} | アシスト={state['assist']:.2f} | "
                  f"入力密度={state['input_density']:.1f} {cooling_mark}")
            print(f"     维持コスト累計: {state['maintenance_cost']:.1f}")
            print(f"     → {state['text']}\n")

        print("=== デモ終了 ===")
        print(f"最終テンション维持コスト: {self.tension_maintenance_cost:.1f}")
        print("安全設計が機能しているか確認できる状態に改善しました。")


if __name__ == "__main__":
    demo = SafeEmotionVGE(seed=777)
    demo.run_demo(28)