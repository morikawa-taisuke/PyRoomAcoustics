import wave as wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as sp
import scipy as scipy
import random
import os
#my_module
from mymodule import my_func, rec_config as rec_conf, rec_utility as rec_util


def serch_reverbe_sec(reverbe_sec, channel):
    reverbe = reverbe_sec
    cnt = 0
    room_dim = np.r_[10.0, 10.0, 10.0]
    """ 音源の読み込み """
    target_data = rec_util.load_wave_data(f'./wave/sample_data/speech/JA/test/JA04F085.wav')
    noise_data = target_data
    wave_data = []  # 1つの配列に格納
    wave_data.append(target_data)
    wave_data.append(noise_data)
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = rec_util.set_mic_coordinate(center=mic_center,
                                               num_channels=num_channels,
                                               distance=distance)  # 各マイクの座標
    doas = np.array([
        [np.pi / 2., np.pi / 2],
        [np.pi / 2., np.pi]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [2., 3.]  # 音源とマイクの距離(m)

    while True:
        e_absorption, max_order = pa.inverse_sabine(reverbe, room_dim)  # Sabineの残響式から壁の吸収率と反射上限回数を決定
        room = pa.ShoeBox(room_dim, fs=rec_conf.sampling_rate, max_order=max_order, absorption=e_absorption)

        """ 部屋にマイクを設置 """
        room.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))
        """ 各音源の座標 """
        source_codinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)
        """ 各音源を部屋に追加する """
        for idx in range(2):
            wave_data[idx] /= np.std(wave_data[idx])
            room.add_source(source_codinate[:, idx], signal=wave_data[idx])

        room.simulate()
        rt60 = room.measure_rt60()
        round_rt60 = round(np.mean(rt60), 3)
        if round_rt60 >= reverbe_sec or cnt == 100:
            break
        cnt+=1
        reverbe+=0.01
        # print(f'[{cnt}]rt60:{np.mean(rt60)}')
    # print(f'max_order:{max_order}\ne_absorption:{e_absorption}')
    print(f'rt60={np.mean(rt60)}')
    return e_absorption, max_order

def calculate_steering_vector(mic_alignments,source_locations,freqs,sound_speed=340,is_use_far=False):
    """ ステアリングベクトルを計算する

    Args:
        mic_alignments: マイクの座標 [[x,y,z],[x,y,z], ...]
        source_locations: 音源の座標 [[x,y,z],[x,y,z], ...]
        freqs:
        sound_speed: 音速 [m/s]
        is_use_far: 音場の仮定

    Returns:
        steering_vector: ステアリングベクトル
    """

    n_channels=np.shape(mic_alignments)[1]  # マイク数を取得
    n_source=np.shape(source_locations)[1]  # 音源数を取得

    if is_use_far==True:    # far-fildの場合
        """ 音源位置を正規化 """
        norm_source_locations=source_locations/np.linalg.norm(source_locations,2,axis=0,keepdims=True)
        """ 位相を求める """
        steering_phase=np.einsum('k,ism,ism->ksm',2.j*np.pi/sound_speed*freqs,norm_source_locations[...,None],mic_alignments[:,None,:])
        """ ステアリングベクトルを算出 """
        steering_vector=1./np.sqrt(n_channels)*np.exp(steering_phase)
        return steering_vector

    else:   # near-fild
        """ 音源とマイクの距離を求める """
        distance=np.sqrt(np.sum(np.square(source_locations[...,None]-mic_alignments[:,None,:]),axis=0))
        """ 遅延時間(delay) [sec] """
        delay=distance/sound_speed
        """ ステアリングベクトルの位相を求める """
        steering_phase=np.einsum('k,sm->ksm',-2.j*np.pi*freqs,delay)
        """ 音量の減衰 """
        steering_decay_ratio=1./distance
        """ ステアリングベクトルを求める """
        steering_vector=steering_decay_ratio[None,...]*np.exp(steering_phase)
        """ 大きさを1で正規化する """
        steering_vector=steering_vector/np.linalg.norm(steering_vector,2,axis=2,keepdims=True)

    return steering_vector


def execute_dsbf(x, a):
    """ 遅延和アレイ

    Args:
        x: 入力信号 [M, Nk, Lt]
        a: ステアリングベクトル [Nk, M]

    Returns:
        c_hat: 出力信号 [M, Nk, Lt]
    """
    """ 遅延和アレイを実行する """
    s_hat = np.einsum("km,mkt->kt", np.conjugate(a), x)
    """ ステアリングベクトルをかける """
    c_hat = np.einsum("kt,km->mkt", s_hat, a)
    return c_hat

def execute_mvdr(x, y, a):
    """MVDR

    Args:
        x: 入力信号
        y: 共分散行列
        a: ステアリングベクトル

    Returns:
        c_hat: 出力信号
    """
    """ 共分散行列を計算する """
    Rcov=np.einsum("mkt,nkt->kmn",y,np.conjugate(y))
    """ 共分散行列の逆行列を計算する """
    Rcov_inverse=np.linalg.pinv(Rcov)
    """ フィルタを計算する """
    Rcov_inverse_a=np.einsum("kmn,kn->km",Rcov_inverse,a)
    a_H_Rcov_inverse_a=np.einsum("kn,kn->k",np.conjugate(a),Rcov_inverse_a)
    w_mvdr=Rcov_inverse_a/np.maximum(a_H_Rcov_inverse_a,1.e-18)[:,None]
    """ フィルタをかける """
    s_hat=np.einsum("km,mkt->kt",np.conjugate(w_mvdr),x)
    """ ステアリングベクトルをかける """
    c_hat=np.einsum("kt,km->mkt",s_hat,a)
    
    return c_hat

def execute_max_snr(x, y):
    """ MAX-SNR

    Args:
        x: 入力信号
        y: 共分散行列

    Returns:
        c_hat: 出力信号
    """
   
    """ 雑音の共分散行列 freq,mic,mic """
    Rn=np.average(np.einsum("mkt,nkt->ktmn",y,np.conjugate(y)),axis=1)
    """ 入力共分散行列 """
    Rs=np.average(np.einsum("mkt,nkt->ktmn",x,np.conjugate(x)),axis=1)
    """ 周波数の数を取得 """
    Nk=np.shape(Rs)[0]
    """ 一般化固有値分解 """
    max_snr_filter=None
    for k in range(Nk):
        w,v=scipy.linalg.eigh(Rs[k,...],Rn[k,...])
        if max_snr_filter is None:
            max_snr_filter=v[None,:,-1]
        else:
            max_snr_filter=np.concatenate((max_snr_filter,v[None,:,-1]),axis=0)
    

    Rs_w=np.einsum("kmn,kn->km",Rs,max_snr_filter)
    beta=Rs_w/np.einsum("km,km->k",np.conjugate(max_snr_filter),Rs_w)[:,None]
    w_max_snr=beta[:,None,:]*max_snr_filter[...,None]
    """ フィルタをかける """
    c_hat=np.einsum("kim,ikt->mkt",np.conjugate(w_max_snr),x)
    
    return(c_hat)

def execute_mwf(x, y, mu):
    """ MWF

    Args:
        x: 入力信号
        y: 共分散行列
        mu: 雑音の共分散行列

    Returns:
        c_hat: 出力信号
    """
   
    """ 雑音の共分散行列 freq,mic,mic """
    Rn=np.average(np.einsum("mkt,nkt->ktmn",y,np.conjugate(y)),axis=1)
    """ 入力共分散行列 """
    Rs=np.average(np.einsum("mkt,nkt->ktmn",x,np.conjugate(x)),axis=1)
    """ 固有値分解をして半正定行列に変換 """
    w,v=np.linalg.eigh(Rs)
    Rs_org=Rs.copy()
    w[np.real(w)<0]=0
    Rs=np.einsum("kmi,ki,kni->kmn",v,w,np.conjugate(v))
    """ 入力共分散行列 """
    Rs_muRn=Rs+Rn*mu
    invRs_muRn=np.linalg.pinv(Rs_muRn)
    """ フィルタ生成 """
    W_mwf=np.einsum("kmi,kin->kmn",invRs_muRn,Rs)
    """ フィルタをかける """
    c_hat=np.einsum("kim,ikt->mkt",np.conjugate(W_mwf),x)
    return c_hat


def calculate_snr(desired, out):
    """ SNRの測定

    Args:
        desired: 目的信号
        out: 処理後の音声

    Returns:
        snr: SNR
    """
    wave_length=np.minimum(np.shape(desired)[0],np.shape(out)[0])

    #消し残った雑音
    desired=desired[:wave_length]
    out=out[:wave_length]
    noise=desired-out
    snr=10.*np.log10(np.sum(np.square(desired))/np.sum(np.square(noise)))

    return(snr)


def Beamforming(result_wave, file_name, out_dir, steering_vectors, noise_only):
    """ すべてのBFを適用

    Args:
        result_wave: シミュレーションにより作成した音声
        file_name: ファイル名
        out_dir: 出力先のディレクトリ
        steering_vectors: ステアリングベクトル
        noise_only: 雑音区間

    Returns:
        None
    """

    """ 短時間フーリエ変換を行う """
    f, t, stft = sp.stft(result_wave, fs=rec_conf.sampling_rate, window="hann", nperseg=rec_conf.frame)
    """ 雑音だけの区間のフレーム数 """
    noise_frame = np.sum(t < (noise_only/rec_conf.sampling_rate))
    """ 雑音だけのデータ """
    stft_noise = stft[..., :noise_frame]
    """MWFの雑音の倍率"""
    mu = 1.0

    """ それぞれのフィルタを実行する """
    dsbf_out = execute_dsbf(stft, steering_vectors[:, 0, :])
    mvdr_out = execute_mvdr(stft, stft, steering_vectors[:, 0, :])
    mlbf_out = execute_mvdr(stft, stft_noise, steering_vectors[:, 0, :])
    max_snr_out = execute_max_snr(stft, stft_noise)
    mwf_out = execute_mwf(stft, stft_noise, mu)

    """ 評価するマイクロホン """
    mix_index = 0

    """ 時間領域の波形に戻す """
    t, dsbf_out = sp.istft(dsbf_out[mix_index], fs=rec_conf.sampling_rate, window="hann", nperseg=rec_conf.frame)
    t, mvdr_out = sp.istft(mvdr_out[mix_index], fs=rec_conf.sampling_rate, window="hann", nperseg=rec_conf.frame)
    t, mlbf_out = sp.istft(mlbf_out[mix_index], fs=rec_conf.sampling_rate, window="hann", nperseg=rec_conf.frame)
    t, max_snr_out = sp.istft(max_snr_out[mix_index], fs=rec_conf.sampling_rate, window="hann", nperseg=rec_conf.frame)
    t, mwf_out = sp.istft(mwf_out[mix_index], fs=rec_conf.sampling_rate, window="hann", nperseg=rec_conf.frame)

    """ ファイルに書き込む """
    """ DS """
    out_path = f"{out_dir}/BF/dsbf/dsbf_{file_name}"
    # print(file_name)
    rec_util.save_wave(dsbf_out[noise_only:] * np.iinfo(np.int16).max / 20.,
                       out_path, rec_conf.sampling_rate)
    """ MVDR """
    out_path = f"{out_dir}/BF/mvdr/mvdr_{file_name}"
    # print(file_name)
    rec_util.save_wave(mvdr_out[noise_only:] * np.iinfo(np.int16).max / 20.,
                       out_path, rec_conf.sampling_rate)
    """ MLBF """
    out_path = f"{out_dir}/BF/mlbf/mlbf_{file_name}"
    # print(file_name)
    rec_util.save_wave(mlbf_out[noise_only:] * np.iinfo(np.int16).max / 20.,
                       out_path, rec_conf.sampling_rate)
    """ MAX_SNR """
    out_path = f"{out_dir}/BF/max_snr/max_snr_{file_name}"
    # print(file_name)
    rec_util.save_wave(max_snr_out[noise_only:] * np.iinfo(np.int16).max / 20.,
                       out_path, rec_conf.sampling_rate)
    """ MWF """
    out_path = f"{out_dir}/BF/mwf/mwf_{file_name}"
    # print(file_name)
    rec_util.save_wave(mwf_out[noise_only:] * np.iinfo(np.int16).max / 20.,
                       out_path, rec_conf.sampling_rate)


def main(clean, snr, out_dir):
    """ すべてのBFを実行
    
    Args:
        clean: 音源
        snr: SNR
        out_dir: 出力先のディレクトリ

    Returns:
        None
    """
    #乱数の種を初期化
    np.random.seed(0)
    noise = './wave/sample_data/noise/hoth.wav'
    #畳み込みに用いる音声波形
    clean_wave_files=[]
    clean_wave_files.append(clean)
    clean_wave_files.append(noise)


    #雑音だけの区間のサンプル数を設定
    n_noise_only=40000

    #音源数
    n_sources=len(clean_wave_files)

    #長さを調べる
    wav=wave.open(clean_wave_files[0])
    n_samples=wav.getnframes()
    wav.close()

    clean_data=np.zeros([n_sources,n_samples+n_noise_only])

    #ファイルを読み込む
    """ 音源の読み込み """
    target_data = rec_util.load_wave_data(clean_wave_files[0])
    noise_data = rec_util.load_wave_data(clean_wave_files[1])
    """ 雑音データをランダムに切り出す """
    start = random.randint(0, len(noise_data) - (len(target_data) + 40000))  # スタート位置をランダムに決定
    noise_data = noise_data[start: start + (len(target_data) + 40000)]  # noise_dataを切り出す
    # print(f'len(target_data):{len(target_data)}')                         # 確認用
    # print(f'len(noise_data):{len(noise_data)}')                           # 確認用
    scale_nosie = rec_util.get_scale_noise(target_data, noise_data, snr)  # noise_dataの大きさを調節
    """ 音源を追加 """
    clean_data[0, n_noise_only:] = target_data
    clean_data[1, :] = scale_nosie

    """ 出力ファイル名を作成 """
    file_name = rec_util.get_file_name(clean)
    noise_name = rec_util.get_file_name(noise)
    file_name = file_name + '_' + noise_name + '.wav'
    print(f'file_name:{file_name}')

    """ 音源のパラメータ """
    sample_rate = rec_conf.sampling_rate    # サンプリング周波数
    N = rec_conf.num_frame                  # フレームサイズ
    Nk = rec_conf.Nk                        # 周波数の数
    freqs = rec_conf.freqs                  # 各ビン数
    """ シミュレーションのパラメータ """
    room_dim = np.r_[10.0, 10.0, 10.0]  # 部屋の大きさ[x,y,z](m)
    num_sources = len(clean_data)  # シミュレーションで用いる音源数
    reverberation = 0.5  # 残響時間(sec)
    e_absorption, max_order = pa.inverse_sabine(reverberation, room_dim)  # Sabineの残響式から壁の吸収率と反射上限回数を決定
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = 4  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = rec_util.set_mic_coordinate(center=mic_center, num_channels=num_channels,
                                               distance=distance)  # 各マイクの座標
    # doas = np.array([
    #     [np.pi / 2., 0],
    #     [np.pi / 2., np.pi]
    # ])                        # 音源の到来方向の指定
    doas = np.array([
        [np.pi / 2., 0],
        [np.pi / 2., np.pi]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [2., 3.]  # 音源とマイクの距離(m)

    # 部屋を生成する
    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=max_order, absorption=e_absorption)
    room_no_noise = pa.ShoeBox(room_dim, fs=sample_rate, max_order=max_order, absorption=e_absorption)

    # 用いるマイクロホンアレイの情報を設定する
    room.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))
    room_no_noise.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))

    """ 各音源の座標 """
    source_codinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    """ 各音源を部屋に追加する """
    for idx in range(num_sources):
        clean_data[idx] /= np.std(clean_data[idx])
        room.add_source(source_codinate[:, idx], signal=clean_data[idx])
        if idx == 0:
            room_no_noise.add_source(source_codinate[:, idx], signal=clean_data[idx])

    #シミュレーションを回す
    room.simulate(snr=100)
    room_no_noise.simulate(snr=100)

    #畳み込んだ波形を取得する(チャンネル、サンプル）
    multi_conv_data=room.mic_array.signals
    multi_conv_data_no_noise=room_no_noise.mic_array.signals

    # #畳み込んだ波形をファイルに書き込む
    # rec_util.save_wave(multi_conv_data_no_noise[0,n_noise_only:]*np.iinfo(np.int16).max/20.,"./mwf_clean.wav",sample_rate)
    # #畳み込んだ波形をファイルに書き込む
    # rec_util.save_wave(multi_conv_data[0,n_noise_only:]*np.iinfo(np.int16).max/20.,"./mwf_in.wav",sample_rate)
    """ チャンネルごとにファイルを分けて保存する """
    for i in range(num_channels):
        out_path = out_dir + f"/reverberation_only/{i}ch/{file_name}"
        #print(file_name)
        rec_util.save_wave(multi_conv_data_no_noise[i, n_noise_only:] * np.iinfo(np.int16).max / 20.,
                                    out_path, sample_rate)
        out_path = out_dir + f"/noise_reverberation/{i}ch/{file_name}"
        # print(file_name)
        rec_util.save_wave(multi_conv_data[i, n_noise_only:] * np.iinfo(np.int16).max / 20.,
                                    out_path, sample_rate)



    #Near仮定に基づくステアリングベクトルを計算: steering_vectors(Nk x Ns x M)
    near_steering_vectors=calculate_steering_vector(mic_codinate, source_codinate, freqs, is_use_far=False)

    #短時間フーリエ変換を行う
    f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

    #雑音だけの区間のフレーム数
    n_noise_only_frame=np.sum(t<(n_noise_only/sample_rate))

    #雑音だけのデータ
    noise_data=stft_data[...,:n_noise_only_frame]

    #MWFの雑音の倍率
    mu=1.0

    #それぞれのフィルタを実行する
    dsbf_out=execute_dsbf(stft_data,near_steering_vectors[:,0,:])
    mvdr_out=execute_mvdr(stft_data,stft_data,near_steering_vectors[:,0,:])
    mlbf_out=execute_mvdr(stft_data,noise_data,near_steering_vectors[:,0,:])
    max_snr_out=execute_max_snr(stft_data,noise_data)
    mwf_out=execute_mwf(stft_data,noise_data,mu)

    #評価するマイクロホン
    eval_mic_index=0

    #時間領域の波形に戻す
    t,dsbf_out=sp.istft(dsbf_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
    t,mvdr_out=sp.istft(mvdr_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
    t,mlbf_out=sp.istft(mlbf_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
    t,max_snr_out=sp.istft(max_snr_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
    t,mwf_out=sp.istft(mwf_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)

    """ SNRをはかる """
    # snr_pre=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],multi_conv_data[eval_mic_index,n_noise_only:])
    # snr_dsbf_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],dsbf_out[n_noise_only:])
    # snr_mvdr_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],mvdr_out[n_noise_only:])
    # snr_mlbf_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],mlbf_out[n_noise_only:])
    # snr_max_snr_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],max_snr_out[n_noise_only:])
    # snr_mwf_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],mwf_out[n_noise_only:])

    """ ファイルに書き込む """
    # rec_util.save_wave(multi_conv_data[eval_mic_index,n_noise_only:]*np.iinfo(np.int16).max/20.,"./mix.wav",sample_rate)
    # rec_util.save_wave(multi_conv_data_no_noise[eval_mic_index,n_noise_only:]*np.iinfo(np.int16).max/20.,"./desired.wav",sample_rate)
    # rec_util.save_wave(dsbf_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./dsbf_out.wav",sample_rate)
    # rec_util.save_wave(mvdr_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./mvdr_out.wav",sample_rate)
    # rec_util.save_wave(mlbf_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./mlbf_out.wav",sample_rate)
    # rec_util.save_wave(max_snr_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./max_snr_out.wav",sample_rate)
    # rec_util.save_wave(mwf_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./mwf_out.wav",sample_rate)

    """ チャンネルごとにファイルを分けて保存する """
    """ DS """
    out_path = out_dir + f"/BF/dsbf/dsbf_{file_name}"
    # print(file_name)
    rec_util.save_wave(dsbf_out[n_noise_only:] * np.iinfo(np.int16).max / 20.,
                                out_path, sample_rate)
    """ MVDR """
    out_path = out_dir + f"/BF/mvdr/mvdr_{file_name}"
    # print(file_name)
    rec_util.save_wave(mvdr_out[n_noise_only:] * np.iinfo(np.int16).max / 20.,
                                out_path, sample_rate)
    """ MLBF """
    out_path = out_dir + f"/BF/mlbf/mlbf_{file_name}"
    # print(file_name)
    rec_util.save_wave(mlbf_out[n_noise_only:] * np.iinfo(np.int16).max / 20.,
                                out_path, sample_rate)
    """ MAX_SNR """
    out_path = out_dir + f"/BF/max_snr/max_snr_{file_name}"
    # print(file_name)
    rec_util.save_wave(max_snr_out[n_noise_only:] * np.iinfo(np.int16).max / 20.,
                                out_path, sample_rate)
    """ MWF """
    out_path = out_dir + f"/BF/mwf/mwf_{file_name}"
    # print(file_name)
    rec_util.save_wave(mwf_out[n_noise_only:] * np.iinfo(np.int16).max / 20.,
                                out_path, sample_rate)


    # print("method:    ", "DSBF", "MVDR", "MLBF", "MaxSNR", "MWF")

    # print("Δsnr [dB]: {:.2f} {:.2f} {:.2f} {:.2f}   {:.2f}".format(snr_dsbf_post-snr_pre,snr_mvdr_post-snr_pre,snr_mlbf_post-snr_pre,snr_max_snr_post-snr_pre,snr_mwf_post-snr_pre))


def main2(wave_files, out_dir, snr, reverbe_sec, reverbe_par, channel=1, is_split=False, rec_type='mix'):
    """ すべてのBFを実行

    Args:
        clean: 音源
        snr: SNR
        out_dir: 出力先のディレクトリ

    Returns:
        None
    """

    """ ファイル名の取得 """
    signal_name = rec_util.get_file_name(wave_files[0])
    noise_name = rec_util.get_file_name(wave_files[1])
    print(f'signal_name:{signal_name}')
    print(f'noise_name:{noise_name}')

    """ 音源の読み込み """
    target_data = rec_util.load_wave_data(wave_files[0])
    noise_data = rec_util.load_wave_data(wave_files[1])

    np.random.seed(0)   # 乱数の種を初期化
    noise_only = 40000  # 雑音だけの区間のサンプル数を設定
    wave_data = np.zeros([len(wave_files), noise_only + len(target_data)])

    """ 雑音データをランダムに切り出す """
    start = random.randint(0, len(noise_data) - (len(target_data) + 40000))  # スタート位置をランダムに決定
    noise_data = noise_data[start: start + (len(target_data) + 40000)]  # noise_dataを切り出す
    # print(f'len(target_data):{len(target_data)}')                         # 確認用
    # print(f'len(noise_data):{len(noise_data)}')                           # 確認用
    scale_nosie = rec_util.get_scale_noise(target_data, noise_data, snr)  # noise_dataの大きさを調節
    """ 音源を追加 """
    wave_data[0, noise_only:] = target_data
    wave_data[1, :] = scale_nosie

    """ 音源のパラメータ """
    sample_rate = rec_conf.sampling_rate  # サンプリング周波数
    N = rec_conf.frame  # フレームサイズ
    Nk = rec_conf.Nk  # 周波数の数
    freqs = rec_conf.freqs  # 各ビン数
    """ シミュレーションのパラメータ """
    room_dim = np.r_[10.0, 10.0, 10.0]  # 部屋の大きさ[x,y,z](m)
    num_sources = len(wave_data)  # シミュレーションで用いる音源数
    mic_center = room_dim / 2  # アレイマイクの中心[x,y,z](m)
    num_channels = channel  # マイクの個数(チャンネル数)
    distance = 0.1  # 各マイクの間隔(m)
    mic_codinate = rec_util.set_mic_coordinate(center=mic_center,
                                               num_channels=num_channels,
                                               distance=distance)  # 各マイクの座標
    doas = np.array([
        [np.pi / 2., 0],
        [np.pi / 2., np.pi]
    ])  # 音源の方向[仰角, 方位角](ラジアン)
    distance = [2., 3.]  # 音源とマイクの距離(m)

    """ 部屋を生成する """
    if rec_type == 'mix':
        rec_type_dir = f'./{out_dir}/mix'
        file_name = f'{signal_name}_{noise_name}_{snr}db_{reverbe_sec * 10}sec.wav'
        room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])
    elif rec_type == 'reverbe':
        rec_type_dir = f'./{out_dir}/reverbe'
        file_name = f'{signal_name}_{reverbe_sec * 10}sec.wav'
        room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=reverbe_par[1], absorption=reverbe_par[0])
    elif rec_type == 'noise':
        rec_type_dir = f'./{out_dir}/noise'
        file_name = f'{signal_name}_{noise_name}_{snr}db.wav'
        room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)
    else :
        rec_type_dir = f'./{out_dir}/clean'
        file_name = f'{signal_name}.wav'
        room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)
    clean_dir = f'./{out_dir}/clean'
    clean_name = f'{signal_name}.wav'
    room_clean = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0, absorption=1.0)

    """ 部屋にマイクを設置する """
    room.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room.fs))
    room_clean.add_microphone_array(pa.MicrophoneArray(mic_codinate, fs=room_clean.fs))

    """ 各音源の座標 """
    source_codinate = rec_util.set_souces_coordinate2(doas, distance, mic_center)

    """ 各音源を部屋に追加する """
    for idx in range(num_sources):
        wave_data[idx] /= np.std(wave_data[idx])
        room.add_source(source_codinate[:, idx], signal=wave_data[idx])
        if idx == 0:
            room_clean.add_source(source_codinate[:, idx], signal=wave_data[idx])

    """ シミュレーションを回す """
    room.simulate()
    room_clean.simulate()

    """ 畳み込んだ波形を取得する(チャンネル、サンプル）"""
    result_wave = room.mic_array.signals
    result_clean = room_clean.mic_array.signals


    if is_split:
        """チャンネルごとにファイルを分けて保存する"""
        for i in range(num_channels):
            """ rec_type """
            out_path = f'{rec_type_dir}_split/{i+1:02}ch/{i+1:02}ch_{file_name}'
            rec_util.save_wave(result_wave[i, :] * np.iinfo(np.int16).max / 20.,
                               out_path, sample_rate)
            """ clean """
            clean_path = f'./{clean_dir}_split/{i + 1:02}ch/{i + 1:02}ch_{clean_name}'
            rec_util.save_wave(result_clean[i, :] * np.iinfo(np.int16).max / 20.,
                               clean_path, sample_rate)
    else:
        """ チャンネルをまとめて保存 """
        """ rec_type """
        out_path = f'{rec_type_dir}/{file_name}'
        result_mix = result_wave * np.iinfo(np.int16).max / 15  # スケーリング
        # print(f'result_mix.shape:{result_mix.shape}')
        rec_util.save_wave(result_mix, out_path)  # 保存
        """ clean """
        clean_path = f'./{clean_dir}/clean/{clean_name}'
        result_clean = result_clean * np.iinfo(np.int16).max / 15  # 全てのチャンネルを保存
        # print(f'result_clean.shape:{result_clean.shape}')               # 確認用
        rec_util.save_wave(result_clean, clean_path)  # 保存

    rt60 = np.mean(room.measure_rt60())
    print(f'{rec_type}：{rt60}')

    """ シミュレーションの条件を記録 """
    text_path = f'{out_dir}/Experimental_conditions.txt'
    with open(text_path, 'a') as out_file:
        out_file.write(f'room_dim:{room_dim}\n'
                       f'mic_codinate:\n{mic_codinate.T}\n'
                       f'source_codinate:\n{source_codinate.T}\n'
                       f'snr:{snr}\n'
                       f'reverbe_sec:{reverbe_sec}\n'
                       f'Rt60:{rt60}\n\n')



    """ Near仮定に基づくステアリングベクトルを計算: steering_vectors(Nk x Ns x M) """
    steering_vectors = calculate_steering_vector(mic_codinate, source_codinate, freqs, is_use_far=False)

    """ Beamformingの適用と出力 """
    Beamforming(result_wave=result_wave,
                file_name=file_name,
                out_dir=out_dir,
                steering_vectors=steering_vectors,
                noise_only=noise_only)


if __name__ == '__main__':
    """ 音声ファイル関係の指定 """
    signal_dir = './wave/sample_data/speech/JA/training'
    noise_dir = './wave/sample_data/noise/hoth.wav'
    signal_list = my_func.get_wave_filelist(signal_dir)
    noise_list = my_func.get_wave_filelist(noise_dir)
    out_dir= './rec_0102'

    """ 録音の条件 """
    snr = 10
    reverbe_sec = 0.5
    channnel = 4
    is_split = False
    reverbe_par = serch_reverbe_sec(reverbe_sec=reverbe_sec, channel=channnel)
    rec_type_list = ['mix']

    """ 出力先の作成 """
    # my_func.exists_dir(my_func.get_dirname(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    """ 音源の条件の記録 """
    text_path = f'{out_dir}/Experimental_conditions.txt'
    with open(text_path, 'w') as text_file:
        text_file.write(f'out_dir:{out_dir}\n'
                       f'signal_dir:{signal_dir}\n'
                       f'noise_list:{noise_list}\n\n')

    for signal in signal_list:
        for noise in noise_list:
            for rec_type in rec_type_list:
                # main(signal, 10, out_dir)

                wave_files = []
                wave_files.append(signal)
                wave_files.append(noise)
                main2(wave_files=wave_files,
                      out_dir=out_dir,
                      snr=snr, reverbe_sec=reverbe_sec, reverbe_par=reverbe_par, channel=channnel,
                      is_split=is_split,
                      rec_type=rec_type)
