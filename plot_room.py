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


def plot_room(channel=1, distance=3, array_type="liner"):
    """ シミュレーションのパラメータ """
    room_dim = np.r_[3.0, 3.0, 3.0]  # 部屋の大きさ[x,y,z](m)
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channel = channel  # マイクの個数(チャンネル数)
    if array_type == "liner":
        mic_coordinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=num_channel, distance=distance*0.01)  # 線形 マイクの座標
    else :
        mic_coordinate = rec_util.set_circular_mic_coordinate(center=mic_center, num_channels=num_channel, radius=distance*0.01/2, rotate=False)   # 円形 マイクの座標

    # mic_coordinate = rec_util.set_circular_mic_coordinate(center=mic_center, num_channels=num_channel, radius=distance * 0.01 / 2, rotate=False)  # 円形 マイクの座標
    # mic_coordinate2 = rec_util.set_circular_mic_coordinate(center=mic_center, num_channels=num_channel, radius=distance * 0.01 / 2, rotate=True)  # 円形 マイクの座標

    doas = np.array([
        [np.pi / 2., np.pi / 2.],  # 話者(音源1)
        [np.pi / 2., np.pi * 0 / 4],  # 雑音(音源2)
        [np.pi / 2., np.pi * 1 / 4],  # 雑音(音源3)
        [np.pi / 2., np.pi * 2 / 4],  # 雑音(音源4)
        [np.pi / 2., np.pi * 3 / 4],  # 雑音(音源5)
        [np.pi / 2., np.pi * 4 / 4],  # 雑音(音源6)

    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [0.5, 0.7, 0.7, 0.7, 0.7, 0.7]  # 音源とマイクの距離(m)
    """ 各音源の座標 """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    print("mic:\n", mic_coordinate)
    print("sound:\n", source_coordinate)
    # fig = plt.figure()
    # ax = fig.add_subplot()

    fig = plt.figure(figsize=(8, 7.2))
    plt.scatter(mic_coordinate[0], mic_coordinate[1], label="Microphones",  marker="d", s=200, edgecolors="b")  # mic
    # plt.scatter(mic_coordinate2[0], mic_coordinate2[1], label="45°",  marker="d", s=300)  # mic
    plt.scatter(source_coordinate[0, 0], source_coordinate[1, 0], label="Speaker", c="red", s=200)    # speaker
    plt.scatter(source_coordinate[0, 1:], source_coordinate[1, 1:], label="Noise", marker="x", c='green', s=200)  # noise
    plt.xlabel("X", fontsize=20) # 軸ラベル
    plt.ylabel("Y", fontsize=20) # 軸ラベル
    plt.xlim(0.5, 2.5)
    plt.ylim(0.7, 2.5)
    plt.xticks([0.5, 1.0, 1.5, 2.0, 2.5], fontsize=18)
    plt.yticks([1.0, 1.5, 2.0, 2.5], fontsize=18)
    # plt.xticks([])
    # plt.yticks([])
    plt.legend(loc="lower left", markerscale=0.75, fontsize="x-large")
    plt.show()



def plot_room_3D(room_dimensions, microphones, sound_sources):
    """
    部屋を3D散布図としてプロットする関数

    Parameters:
        room_dimensions: tuple (x, y, z) 部屋のサイズ
        microphones: list of tuples 各マイクの座標 [(x1, y1, z1), ...]
        sound_sources: list of tuples 各音源の座標 [(x1, y1, z1), ...]
    """
    # 部屋のサイズ
    x_dim, y_dim, z_dim = room_dimensions

    # 3Dプロットのセットアップ
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 部屋の範囲を設定
    ax.set_xlim([1., 2.5])
    ax.set_ylim([1., 2.5])
    ax.set_zlim([1, 2])
    ax.set_xticks([1.0, 1.5, 2.0, 2.5])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax.set_zticks([1.0, 1.5, 2.0])
    ax.set_xlabel('X', fontsize=24, labelpad=30)
    ax.set_ylabel('Y', fontsize=24, labelpad=20)
    ax.set_zlabel('Z', fontsize=24, labelpad=15)

    # マイクをプロット (青)
    for idx, (mic_x, mic_y, mic_z) in enumerate(microphones):
        if idx == 0:
            ax.scatter(mic_x, mic_y, mic_z, marker="d", c='blue', label="Microphones", s=100)
        ax.scatter(mic_x, mic_y, mic_z, marker="d", c='blue', s=100)

    # 音源をプロット (赤)
    src_x, src_y, src_z = zip(sound_sources[0])  # 座標を分解
    ax.scatter(src_x, src_y, src_z, c='red', label='Speaker', s=100)
    # 雑音をプロット (緑)
    src_x, src_y, src_z = zip(sound_sources[1])  # 座標を分解
    ax.scatter(src_x, src_y, src_z, marker="x", c='green', label='Noise', s=100)


    # 凡例を追加
    ax.legend(fontsize="x-large")
    # 各要素の設定をおこなうsetコマンド
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    ax.set_zticklabels(ax.get_zticklabels(), fontsize=20)
    # グリッドを表示
    ax.grid(True)
    # plt.title("3D Room with Microphones and Sound Sources")
    plt.show()


if __name__ == "__main__":
    print("main")

    """録音(シミュレーション)"""
    # plot_room(channel=4, distance=10)

    # Example usage
    # set_circular_mic_coordinate(center=(1.5, 1.5, 1.5), num_channels=4, radius=4)

    """ マイクに関するパラメータ """
    num_mic = 4
    mic_distance = 10   # マイク間隔(cm) 線形：各マイクの間隔, 円形: 円形アレイの直径
    array_type = "liner"    # circular, liner

    """ 部屋に関するパラメータ """
    room_dim = np.array([3, 3, 3])  # 部屋のサイズ (x, y, z)
    angle = 0  # 度 →　ラジアン
    doas = np.array([
        [np.pi / 2., np.pi / 2],    # 目的音声の座標
        [np.pi / 2., np.deg2rad(angle)] # 雑音の座標
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [0.5, 0.7]  # 音源とマイクの距離(m)

    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    mic_coordinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=num_mic, distance=mic_distance*0.01)  # 線形アレイの場合
    # mic_coordinate = rec_util.set_circular_mic_coordinate(center=mic_center, num_channels=num_channels, radius=distance)  # 円形アレイの場合
    """ 各音源の座標 """
    source_coordinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    print(mic_coordinate.T)
    print(source_coordinate.T)
    # plot_room_3D(room_dim, mic_coordinate.T, source_coordinate.T)
    plot_room(channel=num_mic, distance=mic_distance, array_type=array_type)

