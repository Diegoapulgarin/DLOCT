#%%
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import hann, convolve
from scipy.signal import blackman
from scipy.signal import hann
from scipy.interpolate import interp1d
def safe_check(loc_struct, field_name, is_char=False):
    if is_char:
        return loc_struct.get(field_name, '')
    return float(loc_struct.get(field_name, 0))

def read_log_file(file_path):
    if not os.path.isfile(file_path):
        # Asume que file_path es un directorio
        if file_path[-1] in ['/', '\\']:
            file_path = file_path[:-1]
        log_file = os.path.join(file_path, 'Log.xml')
        if not os.path.isfile(log_file):
            # Extraer el nombre de archivo y la extensión del path proporcionado
            base, name_ext = os.path.split(file_path)
            name, ext = os.path.splitext(name_ext)
            log_file = os.path.join(base, name_ext, f"{name_ext}_info.xml")

            if not os.path.isfile(log_file):
                print("Could not open file")
                return None
    else:
        log_file = file_path

    tree = ET.parse(log_file)
    root = tree.getroot()

    if root.tag == 'CCardiOFDISDoc':
        loc = root.find('.//AcqSetting')
        out = {
            'version': 'mfc',
            'numSamples': safe_check(loc, 'numSamples'),
            'numAlinesImage': safe_check(loc, 'numAlinesImage'),
            'numAcqPerImage': safe_check(loc, 'numAcqPerImage'),
            'numAcqFrames': safe_check(loc, 'numAcqFrames'),
            'xGalvoMM': safe_check(loc, 'xGalvoMM'),
            'yGalvoMM': safe_check(loc, 'yGalvoMM'),
            'xGalvoOffset': safe_check(loc, 'xGalvoOffset'),
            'yGalvoOffset': safe_check(loc, 'yGalvoOffset'),
            'pullBackSpeed': root.find('.//m_pbVelText').text,
            'rotSpeed': root.find('.//m_rotVelText').text,
            'm_polygon_reprate': safe_check(root.find('.//m_psArray/Preferences'), 'm_polygon_reprate'),
            'm_dateTime': root.find('.//m_dateTime').text,
            'm_sessionDescription': root.find('.//m_sessionDescription').text,
            'numImages': int(root.find('.//m_numImagesAcquired').text),
        }
        if root.find('.//m_filename2') is not None:
            out['m_filename1'] = root.find('.//m_filename1').text
            out['m_filename2'] = root.find('.//m_filename2').text
        else:
            out['m_filename'] = root.find('.//m_filename1').text if root.find('.//m_filename1') is not None else root.find('.//m_filename').text
        out['m_bkgrnd_filename'] = root.find('.//m_bkgrnd_filename').text
        out['m_log_filename'] = root.find('.//m_log_filename').text
        out['m_ctr1_filename'] = root.find('.//m_ctr1_filename').text
        out['m_ctr2_filename'] = root.find('.//m_ctr2_filename').text
        out['m_AI1_filename'] = root.find('.//m_AI1_filename').text
        out['m_AI2_filename'] = root.find('.//m_AI2_filename').text
        out['m_calib_filename_out'] = root.find('.//m_calib_filename_out').text
    elif root.tag == 'data_info':
        loc = root.find('.//Settings')
        out = {
            'version': 'brian',
            'numSamples': safe_check(loc, 'totalSamplesPerALinePerChannel'),
            'numAlinesImage': safe_check(loc, 'totalALinesPerProcessedBScan'),
            'numImages': safe_check(loc, 'totalRecordedBScans'),
            'nCycles': safe_check(loc.find('recording'), 'rec_autorecordingcycles') if loc is not None else 1,
            'xGalvoMM': safe_check(loc.find('userinterface'), 'ui_galvoxaxisscansizemm') if loc is not None else 0,
            'yGalvoMM': safe_check(loc.find('userinterface'), 'ui_galvoyaxisscansizemm') if loc is not None else 0,
            'xGalvoOffset': safe_check(loc.find('userinterface'), 'ui_galvoxaxisoffsetmm') if loc is not None else 0,
            'yGalvoOffset': safe_check(loc.find('userinterface'), 'ui_galvoyaxisoffsetmm') if loc is not None else 0,
            'yGalvoV2MM': safe_check(loc.find('userinterface'), 'ui_galvovoltagetommconversion') if loc is not None else 0,
            'xyGalvoSwap': safe_check(loc.find('userinterface'), 'ui_scanpatternswapxy') if loc is not None else 0,
            'rotationSpeed': safe_check(loc.find('userinterface'), 'ui_rotationdiallabel%d' % safe_check(loc.find('userinterface'), 'ui_rotationdiallastpos')) if loc is not None else 0,
            'pullbackSpeed': safe_check(loc.find('userinterface'), 'ui_pullbackdiallabel%d' % safe_check(loc.find('userinterface'), 'ui_pullbackdiallastpos')) if loc is not None else 0,
        }
        if out['numImages'] == 0:
            total_alines_per_transfer = safe_check(loc, 'totalALinesPerTransfer')
            if total_alines_per_transfer != 0:
                out['numImages'] = safe_check(loc, 'totalRecordedBlocks') / (out['numAlinesImage'] / total_alines_per_transfer)
            else:
                out['numImages'] = safe_check(loc, 'totalRecordedBlocks')  # Ajuste en caso de división por cero
        file_name = safe_check(loc, 'dataFileName', True) or safe_check(loc, 'dataFilePath', True)
        _, name_ext = os.path.split(file_name)
        name, ext = os.path.splitext(name_ext)
        out['m_filename'] = f"{name}{ext}"
        out['msmtNotes'] = loc.find('Notes').text if loc.find('Notes') is not None else ''
        out['settings'] = safe_check(loc, 'Settings', 1)
    else:
        print("Unknown xml-file version.")
        return None

    return out
def reconstruct_tom_array(data_path, bscan_range=None, parms=None, *args):
    # ReconstructTom reconstructs a tomogram, provided as the name of the folder
    # containing the measurement data
    #
    # reconstructs the tomogram from a measurement, either provided directly as
    # a matrix, together with all other processing relevant parameters, or
    # simply pointed to by path, indicating further a region of interest (roi)
    # indicating the slices to be reconstructed, and optionally processing
    # parameters.
    #
    # Usage:
    # tom = reconstruct_tom(path, roi, parms, unfft) or
    # [S1, S2] = reconstruct_tom(msmt, bgr, map, disp, demodFFTSize, demodulation, sizeOut)
    #
    # where the optional parms dictionary can have the fields
    # parms['bgr'] = {'msmt', 'mean', 'none'}
    # parms['ps'] = {'false', 'true'}
    # parms['map'] = map
    # parms['disp'] = disp
    # parms['window'] = window
    MAX_CHUNK_SIZE = 32 * 1024 ** 2 // 2  # 128 MB, in number of samples

    if isinstance(data_path, str):
        log_data = read_log_file(data_path)
        if log_data is None or not isinstance(log_data, dict):
            raise ValueError('Cannot open Log File')

        n_samples = log_data['numSamples']
        n_alines_per_bscan = log_data['numAlinesImage']
        n_total_bscans = log_data.get('m_numImagesAcquired', log_data.get('numImages'))

        if bscan_range is None:
            bscan_range = [1, n_total_bscans]

        if parms is None:
            parms = {}

        estimate_psf = parms.get('estimatePSF', False)
        use_cal_disp = parms.get('useCalDisp', True)
        bgr = parms.get('bgr', 'msmt')
        ps = parms.get('ps', False)
        aline_step = parms.get('skip', parms.get('alineStep', 1))
        aline_roi = parms.get('alineROI', np.arange(1, n_alines_per_bscan + 1, aline_step))
        n_alines_per_final_bscan = len(aline_roi)
        bscan_step = parms.get('bscanStep', 1)
        map_data, dispersion = read_config(data_path)
        interp_oversamp = parms.get('interpOversamp', 2)
        add_disp = parms.get('addDisp', 1)
        window = hann(len(map_data)) if not parms.get('window') else generate_moving_spectral_window(len(map_data), 0.5, parms['window'])
        n_windows = window.shape[4] if window.ndim == 5 else 1
        spec_trim = parms.get('specTrim', False)
        normalize_spectrum = parms.get('normalizeSpectrum', False)
        zero_mask = parms.get('zeroMask', [])
        size_final_tom = 2 ** int(np.ceil(np.log2(len(map_data)))) if not parms.get('sizeOut') else parms['sizeOut']
        use_complex_fringes = parms.get('useComplexFringes', True)
        demodulate_complex_fringes = parms.get('demodulateComplexFringes', True)
        unfft = parms.get('unfft', False)
        phase_jitter_correction = False
        phase_offset = 0
        jitter_shift = 0

        bgr_data = read_bg(data_path)
        if not isinstance(bgr_data, dict) and bgr == 'msmt':
            bgr = 'mean'

        if not use_cal_disp:
            dispersion = np.ones(len(map_data))

        signal_interp_size = interp_oversamp * n_samples // 2 if use_complex_fringes else interp_oversamp * n_samples
        demod_offset = n_samples // 4

        bscan_batch_size = max(1, round(MAX_CHUNK_SIZE // (n_samples * n_alines_per_final_bscan) * bscan_step))
        if spec_trim:
            spec_trimmed_size = max(np.sum(window != 0, axis=0))
            size_final_tom = 2 * spec_trimmed_size

        tom_shape = (size_final_tom, n_alines_per_final_bscan, (bscan_range[1] - bscan_range[0] + 1) // bscan_step, 2, n_windows)
        tom = np.zeros(tom_shape, dtype=np.float32)
        fringes = np.zeros(tom_shape, dtype=np.float32)
        psf = np.zeros((2, 2), dtype=np.float32)

        last_batch_last_relative_bscan = 0
        for bscan_batch_start_idx in range(bscan_range[0], bscan_range[1], bscan_batch_size):
            this_batch_range = [bscan_batch_start_idx, min(bscan_batch_start_idx + bscan_batch_size - 1, bscan_range[1])]
            signal = read_ofd(data_path, log_data, None, this_batch_range, None, n_alines_per_bscan, bscan_step)

            ch_idx = 0
            for ch in ['ch1', 'ch2']:
                ch_idx += 1
                bg = 'bg1' if ch == 'ch1' else 'bg2'
                this_channel = signal[ch][:, aline_roi - 1, :]
                if bgr == 'msmt':
                    this_background = np.tile(bgr_data[bg], (1, 2))
                elif bgr == 'mean':
                    if ps:
                        b1 = np.mean(signal[ch][:, ::2], axis=1)
                        b2 = np.mean(signal[ch][:, 1::2], axis=1)
                        this_background = np.column_stack((b1, b2))
                    else:
                        this_background = np.tile(np.mean(signal[ch], axis=1, keepdims=True), (1, 2))
                else:
                    this_background = np.zeros((signal[ch].shape[0], 2), dtype=np.float32)

                this_relative_batch_idx = last_batch_last_relative_bscan + np.arange(this_channel.shape[2])

                if phase_jitter_correction:
                    this_phase_offset = phase_offset[:, aline_roi - 1, this_relative_batch_idx]
                    this_jitter_shift = jitter_shift[:, aline_roi - 1, this_relative_batch_idx]
                else:
                    this_phase_offset = 0
                    this_jitter_shift = 0

                tom[:, :, this_relative_batch_idx, ch_idx - 1, :], psf[:, ch_idx - 1], fringes[:, :, this_relative_batch_idx, ch_idx - 1, :] = reconstruct(
                    this_channel, this_background, map_data, dispersion, signal_interp_size, n_samples, demod_offset, add_disp, window, 
                    size_final_tom, unfft, interp_oversamp, zero_mask, estimate_psf, normalize_spectrum, use_complex_fringes, demodulate_complex_fringes, 
                    phase_jitter_correction, this_phase_offset, this_jitter_shift, spec_trim, parms)

            last_batch_last_relative_bscan += len(this_relative_batch_idx)

    else:
        this_channel = data_path
        this_background = np.tile(bscan_range, (1, 2))
        map_data = parms
        dispersion = args[0]
        demod_offset = args[1]
        size_final_tom = args[2]
        add_disp = 1
        interp_oversamp = args[3]
        zero_mask = args[4]
        window = hann(len(map_data)) if len(args) <= 5 or args[5] is None else args[5]
        use_complex_fringes = True if len(args) <= 6 or args[6] is None else args[6]
        demodulate_complex_fringes = True if len(args) <= 7 or args[7] is None else args[7]
        unfft = False
        estimate_psf = False
        n_samples = this_channel.shape[0]
        signal_interp_size = interp_oversamp * n_samples // 2
        normalize_spectrum = False
        spec_trim = False
        phase_jitter_correction = False
        phase_offset = []
        jitter_shift = []

        tom = reconstruct(this_channel, this_background, map_data, dispersion, signal_interp_size, n_samples, demod_offset, add_disp, window, 
                          size_final_tom, unfft, interp_oversamp, zero_mask, estimate_psf, normalize_spectrum, use_complex_fringes, demodulate_complex_fringes, 
                          phase_jitter_correction, phase_offset, jitter_shift, spec_trim, parms)

    if len(args) == 3:
        dim_tom = tom.shape
        dim_tom = (dim_tom[0], dim_tom[1] // 2, dim_tom[2], 3)
        S1 = np.zeros(dim_tom, dtype=np.float32)
        S2 = np.zeros_like(S1)

        S1[..., 0] = np.abs(tom[:, ::2, :, 0]) ** 2 - np.abs(tom[:, ::2, :, 1]) ** 2
        S1[..., 1] = 2 * np.real(tom[:, ::2, :, 0] * np.conj(tom[:, ::2, :, 1]))
        S1[..., 2] = -2 * np.imag(tom[:, ::2, :, 0] * np.conj(tom[:, ::2, :, 1]))

        S2[..., 0] = np.abs(tom[:, 1::2, :, 0]) ** 2 - np.abs(tom[:, 1::2, :, 1]) ** 2
        S2[..., 1] = 2 * np.real(tom[:, 1::2, :, 0] * np.conj(tom[:, 1::2, :, 1]))
        S2[..., 2] = -2 * np.imag(tom[:, 1::2, :, 0] * np.conj(tom[:, 1::2, :, 1]))

        return S1, S2, psf

    elif len(args) == 2:
        return tom, fringes

    elif len(args) == 1:
        return tom

    else:
        return tom, fringes, psf
def get_background(parms, ch, bg, signal, bgr):
    if parms['bgr'] == 'msmt':
        return np.tile(bgr[bg], (1, 2))
    elif parms['bgr'] == 'mean':
        if parms['ps']:
            b1 = np.mean(signal[ch][:, 0::2], axis=1)
            b2 = np.mean(signal[ch][:, 1::2], axis=1)
            return np.stack([b1, b2], axis=-1)
        else:
            return np.tile(np.mean(signal[ch], axis=1), (1, 2))
    elif parms['bgr'] == 'none':
        return np.zeros((signal[ch].shape[0], 2), dtype='float32')
def read_config(file_path, log_struct=None):
    def find_dat_file(directory):
        for item in os.listdir(directory):
            if item.endswith('.dat') and os.path.isfile(os.path.join(directory, item)):
                return os.path.join(directory, item)
        return None

    if log_struct is None:
        if not os.path.isfile(file_path):
            log_struct = read_log_file(file_path)
            if 'm_calib_filename_out' in log_struct:
                config_file = os.path.join(file_path, log_struct['m_calib_filename_out'])
            else:
                config_file = find_dat_file(file_path)
                if config_file is None:
                    print("Can't find the config file.")
                    return None, None
        else:
            config_file = file_path
            file_dir = os.path.dirname(file_path)
            log_struct = read_log_file(file_dir if file_dir else os.getcwd())
    else:
        config_file = file_path

    if not os.path.isfile(config_file):
        print("Can't find the config file.")
        return None, None

    with open(config_file, 'rb') as fid:
        num_samples = log_struct['numSamples']
        map_data = np.fromfile(fid, dtype=np.int32, count=num_samples // 2)
        weight = np.fromfile(fid, dtype=np.float32, count=num_samples // 2)
        map_data = map_data - weight + 1  # +1 to have the first element point to 1

        ddisp = np.fromfile(fid, dtype=np.float32, count=num_samples)
        ddisp = ddisp[:num_samples // 2] - 1j * ddisp[num_samples // 2:]

    return map_data, ddisp
def get_path_for_ext(directory, ext):
    for item in os.listdir(directory):
        if item.endswith(ext) and os.path.isfile(os.path.join(directory, item)):
            return os.path.join(directory, item)
    return None
def read_bg(filename, n_samples):
    bg_filename = filename
    if not os.path.isfile(bg_filename):  # Input is not a filename but a directory name
        directory = filename
        bg_filename = get_path_for_ext(directory, 'ofb')
        if bg_filename is None:
            print("Could not find bg file. Returning 0")
            return {'ch1': 0, 'ch2': 0}

    with open(bg_filename, 'rb') as fid:
        data = np.fromfile(fid, dtype=np.uint16)
    
    n_alines_per_channel = len(data) // n_samples // 2

    if n_alines_per_channel == 1:  # BG recorded has already been averaged
        bg = {
            'ch1': data[:n_samples],
            'ch2': data[n_samples:2*n_samples]
        }
    else:
        ch1 = data[0::2]
        ch2 = data[1::2]
        ch1 = ch1.reshape((n_samples, n_alines_per_channel))
        ch2 = ch2.reshape((n_samples, n_alines_per_channel))
        bg = {
            'ch1': np.mean(ch1, axis=1),
            'ch2': np.mean(ch2, axis=1)
        }

    return bg
def read_ofd(files, log_f=None, n_image=1, bscan_range=None, channels=None, n_alines_per_bscan=None, bscan_step=1):
    BYTES_PER_SAMPLE = 2  # It's always 16 bit
    two_boards = False

    # Check if log is provided, otherwise find and read it
    if log_f is None:
        if isinstance(files, list):  # old format with two boards
            k = max([files[0].rfind(sep) for sep in [os.path.sep, os.path.altsep] if sep])
            log_file = os.path.join(files[0][:k], 'Log.xml')
            log_f = read_log_file(log_file)
        else:
            if not os.path.isfile(files):  # file provided is presumably path
                log_f = read_log_file(files)
            else:  # file name indeed, only log has to be retrieved
                file_dir = os.path.dirname(files)
                log_f = read_log_file(file_dir if file_dir else os.getcwd())

    # Check if files are provided, otherwise find them
    if not isinstance(files, list):
        if not os.path.isfile(files):  # file provided is presumably path
            if 'm_filename1' in log_f:
                msmt_files = [os.path.join(files, log_f['m_filename1']), os.path.join(files, log_f['m_filename2'])]
                two_boards = True
            else:
                if 'm_filename' in log_f:
                    msmt_files = [os.path.join(files, log_f['m_filename'])]
                elif 'dataFilePath' in log_f:
                    base, filename1, filename2 = os.path.splitext(log_f['dataFilePath'])
                    msmt_files = [os.path.join(files, f"{filename1}{filename2}")]
                two_boards = False
        else:  # single file name indeed, only log_f has to be retrieved
            msmt_files = [files]
            file_dir = os.path.dirname(files)
            log_f = read_log_file(file_dir if file_dir else os.getcwd())
            two_boards = False
    else:
        msmt_files = files
        two_boards = len(msmt_files) == 2

    if bscan_range is None:
        bscan_range = [1, log_f['numImages']]

    if n_alines_per_bscan is None:
        n_alines_per_bscan = log_f['numAlinesImage']

    if 'numAcqFrames' not in log_f or log_f['numAcqFrames'] is None:
        log_f['numAcqFrames'] = 1

    n_samples = log_f.get('numSamples', log_f.get('nSamples', None))
    n_bscans_total = bscan_range[1] - bscan_range[0] + 1
    n_bscans_actual = (bscan_range[1] - bscan_range[0] + 1) // bscan_step

    byte_offset = BYTES_PER_SAMPLE * ((bscan_range[0] - 1) * n_samples * n_alines_per_bscan +
                                      (n_image - 1) * n_samples * n_alines_per_bscan * log_f['numAcqFrames'])
    if not two_boards:
        byte_offset *= 2

    fringes = {}
    for file_idx, file in enumerate(msmt_files, start=1):
        with open(file, 'rb') as f:
            f.seek(byte_offset, os.SEEK_SET)
            if two_boards:
                fringes[f'ch{file_idx}'] = np.fromfile(f, dtype=np.uint16, count=n_samples * n_alines_per_bscan * n_bscans_total)
                fringes[f'ch{file_idx}'] = fringes[f'ch{file_idx}'].reshape((n_samples, n_alines_per_bscan, n_bscans_total))
            else:
                if bscan_step == 1:  # Read the whole block
                    temp = np.fromfile(f, dtype=np.uint16, count=2 * n_samples * n_alines_per_bscan * n_bscans_total)
                    fringes['ch1'] = temp[0::2].reshape((n_samples, n_alines_per_bscan, n_bscans_total))
                    fringes['ch2'] = temp[1::2].reshape((n_samples, n_alines_per_bscan, n_bscans_total))
                else:  # Read each Bscan and skip given Bscans
                    fringes['ch1'] = np.zeros((n_samples, n_alines_per_bscan, n_bscans_actual), dtype=np.uint16)
                    fringes['ch2'] = np.zeros_like(fringes['ch1'])
                    this_byte_offset = byte_offset
                    bscan_byte_size = 2 * BYTES_PER_SAMPLE * n_samples * n_alines_per_bscan
                    for this_bscan in range(n_bscans_actual):
                        f.seek(this_byte_offset, os.SEEK_SET)
                        temp = np.fromfile(f, dtype=np.uint16, count=bscan_byte_size // BYTES_PER_SAMPLE)
                        fringes['ch1'][:, :, this_bscan] = temp[0::2].reshape((n_samples, n_alines_per_bscan))
                        fringes['ch2'][:, :, this_bscan] = temp[1::2].reshape((n_samples, n_alines_per_bscan))
                        this_byte_offset += bscan_step * bscan_byte_size

    return fringes
def get_peak_width(data):
    data = data / np.max(data)
    
    psf_max = np.argmax(data)
    psf_range = np.argmin(np.abs(data - np.exp(-2)))
    psf_range = 2 * np.abs(psf_range - psf_max)
    
    first_half_idx = np.arange(psf_max - psf_range, psf_max)
    first_half = data[first_half_idx]
    second_half_idx = np.arange(psf_max, psf_max + psf_range + 1)
    second_half = data[second_half_idx]

    try:
        interp_first_half = interp1d(first_half, first_half_idx, kind='linear')
        interp_second_half = interp1d(second_half, second_half_idx, kind='linear')

        fwhm = interp_second_half(0.5) - interp_first_half(0.5)
        e_sq_radius = interp_second_half(np.exp(-2)) - interp_first_half(np.exp(-2))
    except:
        fwhm = np.nan
        e_sq_radius = np.nan

    return fwhm, e_sq_radius
def reconstruct(signal, background, map_data, disp, signal_interp_size, n_samples, demod_offset, add_disp, window,
                size_final_tom, unfft, interp_oversamp, zero_mask, estimate_psf, normalize_spectrum,
                use_complex_fringes, demodulate_complex_fringes, phase_jitter_correction, phase_offset, jitter_shift, 
                spec_trim, parms):

    def gen_pad_array(array, pad_size, pad_value=0, position='pre'):
        if position == 'pre':
            return np.pad(array, ((pad_size, 0), (0, 0), (0, 0)), 'constant', constant_values=pad_value)
        else:
            return np.pad(array, ((0, pad_size), (0, 0), (0, 0)), 'constant', constant_values=pad_value)

    dim_in = signal.shape
    if len(dim_in) > 2:
        signal = signal.reshape((dim_in[0], np.prod(dim_in[1:])))

    dim = signal.shape
    n_windows = window.shape[4] if window.ndim == 5 else 1

    if use_complex_fringes:
        # Preliminary reconstruction to demodulate, oversampling by using demodFFTSize number of elements
        if dim[1] % 2 == 0:
            pre_tom = fft(signal.astype(np.float32) - np.tile(background.astype(np.float32), (1, dim[1] // 2)), axis=0)
        else:
            pre_tom = fft(signal.astype(np.float32) - background[:, 0:1].astype(np.float32), axis=0)
        
        pre_tom[:n_samples // 2, :, :] = 0
        pre_tom[n_samples // 2 + zero_mask, :, :] = 0
        pre_tom = gen_pad_array(pre_tom, (interp_oversamp - 2) * n_samples // 2, 0, 'pre')

        if demodulate_complex_fringes:
            pre_tom = np.roll(pre_tom, demod_offset, axis=0)
    else:
        if phase_jitter_correction:
            pre_tom = fft(signal.astype(np.float32), axis=0)
            phase_offset = np.exp(1j * phase_offset)
            ramp_fft_shifted = fftshift(np.arange(pre_tom.shape[0]))
            phase_ramp = np.exp(1j * 2 * np.pi * jitter_shift * ramp_fft_shifted[:, None])
            phase_correction = phase_ramp * phase_offset
            pre_tom *= phase_correction
            signal = np.abs(ifft(pre_tom, axis=0))
            if parms['bgr'] == 'mean':
                if background:  # This carries the value from ps
                    b1 = np.mean(signal[:, ::2], axis=1)
                    b2 = np.mean(signal[:, 1::2], axis=1)
                    background = np.stack([b1, b2], axis=1)
                else:
                    background = np.mean(signal, axis=1, keepdims=True).repeat(2, axis=1)
            else:
                background = np.zeros((signal.shape[0], 2))

    map_max = map_data.max()
    if map_max != signal_interp_size:
        map_data = (map_data - 1) / (map_max - 1) * (signal_interp_size - 1) + 1

    if use_complex_fringes:
        kspace_tom = interp1d(np.arange(1, signal_interp_size + 1), ifft(pre_tom, axis=0), axis=0)(map_data)
    else:
        kspace_tom = interp1d(np.arange(1, signal_interp_size + 1), 
                              np.fft.interpft(signal.astype(np.float32) - np.tile(background.astype(np.float32), (1, dim[1] // 2)), signal_interp_size, axis=0), 
                              axis=0)(map_data)

    kspace_tom *= interp_oversamp / 2

    if len(disp) < len(map_data) and len(disp) <= 20:
        disp = np.exp(-1j * np.polyval(disp, np.linspace(-1, 1, len(map_data))))
    elif len(disp) != len(map_data):
        disp = interp1d(np.linspace(-1, 1, len(disp)), disp, kind='linear', fill_value='extrapolate')(np.linspace(-1, 1, len(map_data)))

    if len(add_disp) > 1 and len(add_disp) != len(map_data):
        add_disp = interp1d(np.linspace(-1, 1, len(add_disp)), add_disp, kind='linear', fill_value='extrapolate')(np.linspace(-1, 1, len(map_data)))

    if normalize_spectrum:
        envelope = convolve(np.max(np.abs(kspace_tom), axis=1), hann(41), mode='same')
        envelope = envelope / envelope.max()
        kspace_tom = kspace_tom / envelope[:, None]

    kspace_tom *= window * disp[:, None] * add_disp[:, None]

    if spec_trim:
        spec_trimmed_size = np.max(np.sum(window != 0, axis=0))
        spec_start = np.argmax(window != 0, axis=0)
        kspace_tom_trimmed = np.zeros((spec_trimmed_size, *kspace_tom.shape[1:]), dtype=kspace_tom.dtype)
        for i in range(window.shape[4]):
            kspace_tom_trimmed[:, :, :, :, i] = kspace_tom[spec_start[i]:spec_start[i] + spec_trimmed_size, :, :, :, i]
        kspace_tom = kspace_tom_trimmed

    if not unfft:
        if use_complex_fringes:
            k_dims = kspace_tom.shape
            kspace_tom = np.pad(kspace_tom, ((0, size_final_tom - k_dims[0]), (0, 0), (0, 0), (0, 0), (0, 0)), 'constant')
            kspace_tom = np.roll(kspace_tom, -k_dims[0] // 2, axis=0)
            tom = fft(kspace_tom, axis=0)
            fringes = kspace_tom
            if demodulate_complex_fringes:
                tom = fftshift(tom, axes=0)
        else:
            tom = fftshift(fft(np.roll(np.pad(kspace_tom, ((0, 2 * size_final_tom - len(map_data)), (0, 0), (0, 0), (0, 0), (0, 0)), 'constant'), -len(map_data) // 2, axis=0), axis=0), axes=0)
            fringes = np.pad(kspace_tom, ((0, 2 * size_final_tom - len(map_data)), (0, 0), (0, 0), (0, 0), (0, 0)), 'constant')
    else:
        tom = np.roll(np.pad(kspace_tom, ((0, size_final_tom - len(map_data)), (0, 0)), 'constant'), -len(map_data) // 2, axis=0)
        fringes = np.pad(kspace_tom, ((0, size_final_tom - len(map_data)), (0, 0), (0, 0), (0, 0), (0, 0)), 'constant')

    if dim != dim_in:
        tom_shape = [size_final_tom if use_complex_fringes else 2 * size_final_tom, *dim_in[1:], 1, 1, n_windows]
        tom = tom.reshape(tom_shape)
        fringes = fringes.reshape(tom_shape)

    if estimate_psf:
        size_final_tom_nominal = 2 ** np.ceil(np.log2(len(map_data)))
        interp_factor = round(10 * size_final_tom_nominal / size_final_tom)
        psf_exp = np.abs(fftshift(fft(np.pad(np.mean(np.abs(kspace_tom), axis=1), (0, size_final_tom * interp_factor - kspace_tom.shape[0]), 'constant')))) ** 2
        psf_fwhm, wz = get_peak_width(psf_exp)
        psf_fwhm /= interp_factor
        wz /= interp_factor
        psf = [psf_fwhm, wz]
    else:
        psf = [np.nan, np.nan]

    return tom, psf, fringes
def generate_moving_spectral_window(n_k_samples, spec_overlap, n_spec_vec):
    axial_res_red_vec = n_spec_vec - (n_spec_vec - 1) * spec_overlap
    split_spec_width_vec = np.round(n_k_samples / axial_res_red_vec).astype(int)
    sweep_spec_center = True

    fft_window_mat = np.zeros((n_k_samples, n_spec_vec), dtype=np.float32)

    for ax_rr_idx in range(len(axial_res_red_vec)):
        axial_res_red = axial_res_red_vec
        if sweep_spec_center:
            this_n_spec_vec = np.arange(1, n_spec_vec + 1)
        else:
            this_n_spec_vec = np.array([(1 + n_spec_vec) // 2])

        for spec in this_n_spec_vec:
            this_width = split_spec_width_vec
            this_spec_beg = int(np.round(this_width * (spec - 1) - this_width * spec_overlap * (spec - 1)) + 1)
            this_spec_beg = min(max(this_spec_beg, 1), n_k_samples)
            this_spec_end = int(np.round(this_width * spec - this_width * spec_overlap * (spec - 1)))
            this_spec_end = min(max(this_spec_end, 1), n_k_samples)

            this_width = this_spec_end - this_spec_beg + 1
            this_fft_window = np.zeros(n_k_samples)
            this_fft_window[this_spec_beg - 1:this_spec_end] = blackman(this_width)

            fft_window_mat[:, spec - 1] = this_fft_window

    return fft_window_mat
#%%
# Definir la carpeta de datos
data_folder = r'E:\DLOCT\TomogramsDataAcquisition\[DepthWrap]'

# Definir el nombre del archivo de datos
data_file_name = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'

# Definir los parámetros de reconstrucción
reconstruct_parms = {
    'demodulateComplexFringes': False,
    'useComplexFringes': False,
    'bgr': 'mean',
    'sizeOut': 2048,
    'map': list(range(1, 1281, 2)),
    'disp': 0
}

# Ruta completa al archivo de datos
data_file_path = os.path.join(data_folder, data_file_name)

# Llamar a la función de reconstrucción
tom_hhp = reconstruct_tom_array(data_file_path, [0, 10], reconstruct_parms)

#%%
file_path = data_file_path
if not os.path.isfile(file_path):
    # Asume que file_path es un directorio
    if file_path[-1] in ['/', '\\']:
        file_path = file_path[:-1]
    log_file = os.path.join(file_path, 'Log.xml')
    if not os.path.isfile(log_file):
        # Extraer el nombre de archivo y la extensión del path proporcionado
        base, name_ext = os.path.split(file_path)
        name, ext = os.path.splitext(name_ext)
        log_file = os.path.join(base, name_ext,f"{name_ext}_info.xml")

        if not os.path.isfile(log_file):
            print("Could not open file")