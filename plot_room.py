import matplotlib.pyplot as plt
import pyroomacoustics as pa
import numpy as np

# my_module
import rec_config as rec_conf
import rec_utility as rec_util
from mymodule import my_func


def set_circular_mic_coordinate(center, num_channels, radius):
    """ アレイマイクの各マイクの座標を決める (円形アレイ)
    :param center: マイクの中心点
    :param num_channels: チャンネル数
    :param distance: マイク間の距離
    :return coordinate: マイクの座標
    """
    angle_list = np.linspace(0, 2 * np.pi, num_channels, endpoint=False)
    print(angle_list)
    if len(center) == 2:
        x_points = center[0] + radius * np.cos(angle_list)
        y_points = center[1] + radius * np.sin(angle_list)
        coordinate = np.array([x_points.tolist(), y_points.tolist()])
    else:
        x_points = center[0] + radius * np.cos(angle_list)
        y_points = center[1] + radius * np.sin(angle_list)
        z_points = np.full(num_channels, center[2])
        coordinate = np.array([x_points.tolist(), y_points.tolist(), z_points.tolist()])

    # print(coordinate)

    return coordinate


def plot_room(channel=1):
    """ シミュレーションのパラメータ """
    room_dim = np.r_[10.0, 10.0, 10.0]  # 部屋の大きさ[x,y,z](m)
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.05/2.0  # 各マイクの間隔(m)
    # mic_coordinate = rec_util.set_mic_coordinate(center=mic_center,
    #                                              num_channels=num_channels,
    #                                              distance=distance)  # 各マイクの座標
    mic_coordinate = rec_util.set_circular_mic_coordinate(center=room_dim/2, num_channels=channel, radius=distance)
    doas = np.array([
        [np.pi / 2., np.pi / 2.],  # 話者(音源1)
        [np.pi / 2., np.pi * 0 / 4],  # 雑音(音源2)
        [np.pi / 2., np.pi * 1 / 4],  # 雑音(音源3)
        [np.pi / 2., np.pi * 2 / 4],  # 雑音(音源4)
        [np.pi / 2., np.pi * 3 / 4],  # 雑音(音源5)
        [np.pi / 2., np.pi * 4 / 4],  # 雑音(音源6)

    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [2., 3., 3., 3., 3., 3.]  # 音源とマイクの距離(m)
    """ 各音源の座標 """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    print('mic:\n', mic_coordinate)
    print('sound:\n', source_coordinate)
    # fig = plt.figure()
    # ax = fig.add_subplot()

    plt.scatter(mic_coordinate[0], mic_coordinate[1], label='mic', marker="D", s=200, edgecolors="b")
    plt.scatter(source_coordinate[0, 0], source_coordinate[1, 0], label='speeker', marker="^", s=200)
    plt.scatter(source_coordinate[0, 1:], source_coordinate[1, 1:], label='noise', marker="x", s=200)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend(loc='best', markerscale=0.75)
    plt.show()

    """ 部屋の出力 """
    # fig, ax = room_mix.plot()
    # ax.set_xlim([0, 10])
    # ax.set_ylim([0, 10])
    # ax.set_zlim([0, 10])
    # plt.show()


if __name__ == '__main__':
    print('main')
    # a = np.array([[1., 4.],
    #               [2., 5.],
    #               [3., 6.]])
    # print(a[0, 0], a[1, 0])
    # print(a[0, 1], a[1, 1])

    """録音(シミュレーション)"""
    plot_room(channel=4)

    # ch = 4
    # radius = 1  # 半径
    # angle_list = np.array([[360*i/ch]for i in range(ch)])
    # print(angle_list)
    # angle_list = np.array([[np.cos(2*np.pi*i/ch) * radius, np.sin(2*np.pi*i/ch) * radius]for i in range(ch)])
    # print(angle_list)
    # angle_list = np.array([[np.cos(360*i/ch) * radius, np.sin(360*i/ch) * radius]for i in range(ch)])
    # print(angle_list)
    #



    # Example usage
    # plot_points_on_circle(center=(5, 5), num_points=4, radius=1)
    set_circular_mic_coordinate([5, 5, 5], 4, 1)
