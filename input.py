import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# u1和u3必须一样吗？

class generate_input():

    def __init__(self):
        self.u1_high = 3.8e9
        self.u1_high_adp = 3.4e9
        self.u1_low = 2.8e9
        self.u1_low_adp = 3.2e9

        self.u2_high = 1.9e9
        self.u2_high_adp = 1.5e9
        self.u2_low = 0.9e9
        self.u2_low_adp = 1.3e9

        self.u3_adp = 0.2e9

        self.u1_boundary = 3.3e9
        self.u2_boundary = 1.4e9

        # 高输入保持的上下界
        self.high_step_1 = 500
        self.high_step_2 = 600

        # 低输入保持的上下界
        self.low_step_1 = 200
        self.low_step_2 = 300

        # action noise
        self.an = np.array([10, 10, 10])
        self.bn = np.array([100, 100, 100])

    def signal_generate(self, T):
        signal = np.zeros((T, 3))
        signal_noise = np.zeros((T, 3))
        current_value_0 = np.random.uniform(self.u1_low, self.u1_high)  # 初始值在指定范围内随机生成
        current_value_2 = current_value_0 + np.random.uniform(-self.u3_adp, self.u3_adp)
        if current_value_0 < self.u1_boundary:
            current_value_1 = np.random.uniform(self.u2_low, self.u2_low_adp)
        else:
            current_value_1 = np.random.uniform(self.u2_high_adp, self.u2_high)

        consecutive_low_count = 0

        length = 0

        while length < T:
            if current_value_0 < self.u1_boundary and consecutive_low_count < self.low_step_1:
                current_value_0 = np.random.uniform(self.u1_low, self.u1_low_adp)
                current_value_1 = np.random.uniform(self.u2_low, self.u2_low_adp)
                current_value_2 = current_value_0 + np.random.uniform(-self.u3_adp, self.u3_adp)

            # 生成保持的长度，对于低的情况，step在200到300之间
            if current_value_0 < self.u1_boundary:
                segment_length = np.random.randint(self.low_step_1, self.low_step_2)
            else:
                segment_length = np.random.randint(self.high_step_1, self.high_step_2)
                consecutive_low_count = 0

            # 最后一段直接填充
            if (T - length) < self.high_step_2 + 1:
                segment_length = T - length

            # 添加当前值保持的段
            signal[length: length + segment_length, 0] = np.full(segment_length, current_value_0)
            signal[length: length + segment_length, 1] = np.full(segment_length, current_value_1)
            signal[length: length + segment_length, 2] = np.full(segment_length, current_value_2)

            # 更新连续低的计数
            if current_value_0 < self.u1_boundary:
                consecutive_low_count += segment_length

            # 切换到高或低
            current_value_0 = np.random.uniform(self.u1_low,
                                                self.u1_low_adp) if current_value_0 > self.u1_boundary else np.random.uniform(
                self.u1_high_adp, self.u1_high)
            current_value_1 = np.random.uniform(self.u2_low,
                                                self.u2_low_adp) if current_value_1 > self.u2_boundary else np.random.uniform(
                self.u2_high_adp, self.u2_high)
            current_value_2 = current_value_0 + np.random.uniform(-self.u3_adp, self.u3_adp)

            length += segment_length

        for i in range(0, T):
            signal_noise[i, :] = np.clip(np.random.normal(np.zeros_like(self.an), self.an), -self.bn, self.bn)

        signal += signal_noise
        return torch.tensor(signal)


if __name__ == '__main__':
    T = 5000
    action = generate_input()
    signal = action.signal_generate(T)

    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 3))

    axs[0].plot(signal[:, 0], label='u1', color='red', linewidth=0.8)
    axs[1].plot(signal[:, 1], label='u2', color='red', linewidth=0.8)
    axs[2].plot(signal[:, 2], label='u3', color='red', linewidth=0.8)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig.suptitle('input signal generate')
    plt.show()
