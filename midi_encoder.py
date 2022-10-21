import argparse
import copy
import math as ma
import os
import uuid
from collections import Counter
from multiprocessing import Process

import mido
from miditok.utils import merge_tracks
from mido import MidiFile as MidoFile, MidiTrack
from miditok import REMI, get_midi_programs, MuMIDI, Octuple
from miditoolkit import MidiFile
import music21 as m21
import music21.duration
import numpy as np

from blacklist import blacklist, blacklist_clean, blacklist_old

whitelist = []
THREE_DOTTED_BREVE = 15
THREE_DOTTED_32ND  = 0.21875

MIN_VELOCITY = 0
MAX_VELOCITY = 160

MIN_TEMPO = 24
MAX_TEMPO = 200

MAX_PITCH = 128

FREQ = 16

def remi_multiprocessing_encode(datapath, tokenizer, transpose_range, n_process):
    batched_data = [os.listdir(datapath)[i:i + n_process] for i in range(0, len(os.listdir(datapath)), n_process)]

    for i, batch in enumerate(batched_data):
            threads = []
            for file in batch:
                file_path = os.path.join(datapath, file)
                file_extension = os.path.splitext(file_path)[1]
                if os.path.isfile(file_path) and (file_extension == ".midi" or file_extension == ".mid") and ('decoded' not in file_path):
                    thread = Process(target=remi_encode, args=(file_path, tokenizer, transpose_range))
                    threads.append(thread)
                    thread.start()
            [t.join() for t in threads]

def remi_encode(file_path, tokenizer, transpose_range=0):
    print(f"Processing {file_path}")
    midi = MidiFile(file_path)
    # midi2 = MidiFile(file_path)
    # if len(midi.instruments) > 1:
    #     for i in range(1, len(midi.instruments)):
    #         midi.instruments[0].notes.extend(midi.instruments[i].notes)
    #     midi.instruments = midi.instruments[:1]
    merged_track = merge_tracks(midi, effects=False)
    midi.instruments = [merged_track]

    # Converts MIDI to tokens, and back to a MIDI
    raw_tokens = tokenizer.midi_to_tokens(midi)
    converted_back_midi = tokenizer.tokens_to_midi(raw_tokens)
    file_folder = file_path.split('/')[:-1]
    file_name = file_path.split('/')[-1]
    decoded_path = os.path.join(*file_folder, file_name.split('.mid')[0] + '_decoded.mid')
    converted_back_midi.dump(decoded_path)

    versions = []

    for trans_value in range(int(-transpose_range / 2), int(transpose_range/2) + 1):
        tmp = copy.deepcopy(midi)
        for note in tmp.instruments[0].notes:
            note.pitch += trans_value
        version_int = [tokenizer.vocab.event_to_token['SOS_None']] + tokenizer.midi_to_tokens(tmp)[0] + [tokenizer.vocab.event_to_token['EOS_None']]
        # version_str = tokenizer.tokens_to_events(version_int)
        version_str = [str(t) for t in version_int]
        versions.append(version_str)
    with open(os.path.join(*file_folder, file_name.split('.mid')[0] + '.txt'), mode='w') as token_file:
        versions = [' '.join(v) for v in versions]
        output_txt = "\n".join(versions)
        token_file.write(output_txt)
def process(file_path, file_extension, sample_freq, piano_range, transpose_range, stretching_range):
    if os.path.isfile(file_path) and (file_extension == ".midi" or file_extension == ".mid"):
        parse_midi(file_path, sample_freq, piano_range, transpose_range, stretching_range)


def multiprocessing_encode(datapath, n_process, sample_freq=FREQ, piano_range=(33, 93), transpose_range=10, stretching_range=10):
    if os.path.isfile(datapath):
        # Path is an individual midi file
        file_extension = os.path.splitext(datapath)[1]

        if file_extension == ".midi" or file_extension == ".mid":
            text = parse_midi(datapath, sample_freq, piano_range, transpose_range, stretching_range)
    else:
        # Read every file in the given directory

        batched_data = [os.listdir(datapath)[i:i + n_process] for i in range(0, len(os.listdir(datapath)), n_process)]

        for i, batch in enumerate(batched_data):
            threads = []
            for file in batch:
                file_path = os.path.join(datapath, file)
                file_extension = os.path.splitext(file_path)[1]
                thread = Process(target=process, args=(file_path, file_extension, sample_freq, piano_range, transpose_range, stretching_range))
                threads.append(thread)
                thread.start()
            [t.join() for t in threads]

def load(datapath, sample_freq=FREQ, piano_range=(33, 93), transpose_range=10, stretching_range=10, ignore = False):
    text = ""
    vocab = set()

    if os.path.isfile(datapath):
        # Path is an individual midi file
        file_extension = os.path.splitext(datapath)[1]

        if file_extension == ".midi" or file_extension == ".mid":
            text = parse_midi(datapath, sample_freq, piano_range, transpose_range, stretching_range, ignore)
            vocab = set(text.split(" "))
    else:
        # Read every file in the given directory
        for file in os.listdir(datapath):
            file_path = os.path.join(datapath, file)
            file_extension = os.path.splitext(file_path)[1]

            # Check if it is not a directory and if it has either .midi or .mid extentions
            if os.path.isfile(file_path) and (file_extension == ".midi" or file_extension == ".mid") and ('decoded' not in file_path):
                encoded_midi = parse_midi(file_path, sample_freq, piano_range, transpose_range, stretching_range, ignore)

                if len(encoded_midi) > 0:
                    words = set(encoded_midi.split(" "))
                    vocab = vocab | words

                    text += encoded_midi + " "

        # Remove last space
        text = text[:-1]

    return text, vocab


def parse_midi(file_path, sample_freq, piano_range, transpose_range, stretching_range,ignore = False):
    print("Parsing midi file:", file_path)

    # Split datapath into dir and filename
    midi_dir = os.path.dirname(file_path)
    midi_name = os.path.basename(file_path).split(".")[0]

    # If txt version of the midi already exists, load data from it
    midi_txt_name = os.path.join(midi_dir, midi_name + ".txt")

    if (os.path.isfile(midi_txt_name)):
        midi_fp = open(midi_txt_name, "r")
        encoded_midi = midi_fp.read()
        midi_fp.close()
        return encoded_midi
    else:
        return ""
    #     if ignore:
    #         return ""
    #     random_name = uuid.uuid4().hex + ".mid"
    #     random_path = os.path.join(*file_path.split('/')[:-1], random_name)
    #     m = mido.MidiFile(file_path)
    #     new_merged_tracks = []
    #     for track in m.tracks:
    #         for i in range(len(track)):
    #             if not track[i].is_meta:
    #                 track[i].channel = 0
    #     merged_tracks = mido.merge_tracks(m.tracks)
    #
    #     mido_empty = MidoFile(type=0)
    #     mido_empty.tracks.append(merged_tracks)
    #     mm = merged_tracks[0]
    #     mido_empty.ticks_per_beat = m.ticks_per_beat
    #
    #     mido_empty.save(random_path)
    #     # Create a music21 stream and open the midi file
    #     midi = m21.midi.MidiFile()
    #     midi.open(random_path)
    #     midi.read()
    #     midi.close()
    #
    #     m = m21.converter.parse(random_path)
    #     fp = m.write('midi', fp='pathToWhereYouWantToWriteIt.mid')
    #     # Creates the tokenizer and loads a MIDI
    #     pitch_range = range(21, 109)
    #     beat_res = {(0, 4): 8, (4, 12): 16}
    #     nb_velocities = 32
    #     additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
    #                          'rest_range': (2, 8),  # (half, 8 beats)
    #                          'nb_tempos': 32,  # nb of tempo bins
    #                          'tempo_range': (40, 250)}  # (min, max)
    #     tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=False)
    #
    #     midi = MidiFile(file_path)
    #     # midi2 = MidiFile(file_path)
    #     midi.instruments[1].notes.extend(midi.instruments[0].notes)
    #     midi.instruments = midi.instruments[1:]
    #
    #
    #     # Converts MIDI to tokens, and back to a MIDI
    #     tokens = tokenizer.midi_to_tokens(midi)
    #     converted_back_midi = tokenizer.tokens_to_midi(tokens, get_midi_programs(midi))
    #     converted_back_midi.dump('merged4.mid')
    #     # Translate midi to stream of notes and chords
    #     encoded_midi = midi2encoding(midi, sample_freq, piano_range, transpose_range, stretching_range)
    #     os.remove(random_path)
    #     if len(encoded_midi) > 0:
    #         midi_fp = open(midi_txt_name, "w+")
    #         midi_fp.write(encoded_midi)
    #         midi_fp.flush()
    #         midi_fp.close()
    # return encoded_midi


def midi2encoding(midi, sample_freq, piano_range, transpose_range, stretching_range):
    try:
        # midi.tracks = midi.tracks[1:]
        midi_stream = m21.midi.translate.midiFileToStream(midi)

    except:
        return []

    # Get piano roll from midi stream
    piano_roll = midi2piano_roll(midi_stream, sample_freq, piano_range, transpose_range, stretching_range)

    # Get encoded midi from piano roll
    encoded_midi = list(piano_roll2encoding(piano_roll))

    # if encoded_midi:
    #     d = len(Counter(encoded_midi[0].split()))
    #     write(encoded_midi[0],'resuljt.mid')

    return " ".join(encoded_midi)


def piano_roll2encoding(piano_roll):
    # Transform piano roll into a list of notes in string format
    final_encoding = {}

    perform_i = 0
    for version in piano_roll:
        lastTempo = -1
        lastVelocity = -1
        lastDuration = -1.0

        version_encoding = []

        for i in range(len(version)):
            # Time events are stored at the last row
            tempo = version[i, -1][0]
            if tempo != 0 and tempo != lastTempo:
                version_encoding.append("t_" + str(int(tempo)))
                lastTempo = tempo

            # Process current time step of the piano_roll
            for j in range(len(version[i]) - 1):
                duration = version[i, j][0]
                velocity = int(version[i, j][1])

                if velocity != 0 and velocity != lastVelocity:
                    version_encoding.append("v_" + str(velocity))
                    lastVelocity = velocity

                if duration != 0 and duration != lastDuration:
                    duration_tuple = m21.duration.durationTupleFromQuarterLength(duration)
                    version_encoding.append("d_" + duration_tuple.type + "_" + str(duration_tuple.dots))
                    lastDuration = duration

                if duration != 0 and velocity != 0:
                    version_encoding.append("n_" + str(j))

            # End of time step
            if len(version_encoding) > 0 and version_encoding[-1][0] == "w":
                # Increase wait by one
                # if int(version_encoding[-1].split("_")[1]) > 128:
                #     return ""
                version_encoding[-1] = "w_" + str(int(version_encoding[-1].split("_")[1]) + 1)
            else:
                version_encoding.append("w_1")

        # End of piece
        if version_encoding[-1] != 'w_64' or  version_encoding[-1] != 'w_128':
            version_encoding = version_encoding[:-1]
        version_encoding.append("\n")
        # if any(token in blacklist_old for token in version_encoding):
        #     return ""
        # Check if this version of the MIDI is already added
        version_encoding_str = " ".join(version_encoding)
        if version_encoding_str not in final_encoding:
            final_encoding[version_encoding_str] = perform_i

        perform_i += 1

    return final_encoding.keys()


def write(encoded_midi, path):
    # Base class checks if output path exists
    midi = encoding2midi(encoded_midi)
    midi.open(path, "wb")
    midi.write()
    return midi


def encoding2midi(note_encoding, ts_duration=1/FREQ):
    notes = []

    velocity = 100
    duration = "16th"
    dots = 0

    ts = 0
    for note in note_encoding.split(" "):
        if len(note) == 0:
            continue

        elif note[0] == "w":
            wait_count = int(note.split("_")[1])
            ts += wait_count

        elif note[0] == "n":
            pitch = int(note.split("_")[1])
            note = m21.note.Note(pitch)
            note.duration = m21.duration.Duration(type=duration, dots=dots)
            note.offset = ts * ts_duration
            note.volume.velocity = velocity
            notes.append(note)

        elif note[0] == "d":
            duration = note.split("_")[1]
            dots = int(note.split("_")[2])

        elif note[0] == "v":
            velocity = int(note.split("_")[1])

        elif note[0] == "t":
            tempo = int(note.split("_")[1])

            if tempo > 0:
                mark = m21.tempo.MetronomeMark(number=tempo)
                mark.offset = ts * ts_duration
                notes.append(mark)

    piano = m21.instrument.fromString("Piano")
    notes.insert(0, piano)

    piano_stream = m21.stream.Stream(notes)
    main_stream = m21.stream.Stream([piano_stream])

    return m21.midi.translate.streamToMidiFile(main_stream)


def midi_parse_notes(midi_stream, sample_freq):
    note_events = []
    note_filter = m21.stream.filters.ClassFilter('Note')
    i = 0
    base_quarter_length = midi_stream.parts[0].quarterLength
    measure_offsets = []
    parts = []
    always_use_measure_offset = False
    for part in midi_stream.parts:
        parts.append(part)
    if len(parts) > 2:
        print(123)
    parts = sorted(parts, key=lambda row: row.highestTime, reverse=False)
    if len(parts) > 1:
        if abs(parts[0].highestTime - parts[1].highestTime) > 40:
            for measure in parts[0].measures(0, None):
                measure_offsets.append(measure.offset)
        if len(parts[0].measures(0, None)) == len(parts[1].measures(0, None)):
            always_use_measure_offset = True
    for part in midi_stream.parts:
        for sm in part.semiFlat:
            print(123)
        part.duration.quarterLength = base_quarter_length
        scale = base_quarter_length / part.quarterLength
        if part.offset > 0:
            print("Greater than 0")
        n_mes = 0
        last_measure_offset = part.measures(0, None)[0].offset
        base_measure_offset = part.measures(0, None)[1].offset - part.measures(0, None)[0].offset
        for measure in part.measures(0, None):
            for note in measure.recurse().addFilter(note_filter):
                pitch = note.pitch.midi
                duration = note.duration.quarterLength
                velocity = note.volume.velocity
                if always_use_measure_offset:
                    offset = int((part.offset + measure_offsets[n_mes] + note.offset) * sample_freq)
                else:
                    if abs(measure.offset - last_measure_offset) > base_measure_offset * 2:
                        if measure_offsets:
                            offset = int((part.offset + measure_offsets[n_mes]+ note.offset) * sample_freq)
                        else:
                            return []
                    else:
                        offset = int((part.offset + measure.offset + note.offset) * sample_freq)
                note_events.append((pitch, duration, velocity, int(offset)))
                i+=1
            n_mes+=1
            last_measure_offset = measure.offset
    return note_events


def midi_parse_chords(midi_stream, sample_freq):
    chord_filter = m21.stream.filters.ClassFilter('Chord')
    note_events = []
    base_quarter_length = midi_stream.parts[0].quarterLength
    parts = []
    measure_offsets = []
    for part in midi_stream.parts:
        parts.append(part)
    always_use_measure_offset= False
    parts = sorted(parts, key=lambda row: row.highestTime, reverse=False)
    if len(parts) > 1:
        if abs(parts[0].highestTime - parts[1].highestTime) > 40:
            for measure in parts[0].measures(0, None):
                measure_offsets.append(measure.offset)
        if len(parts[0].measures(0, None)) == len(parts[1].measures(0, None)):
            always_use_measure_offset = True
    for part in midi_stream.parts:
        scale = base_quarter_length / part.quarterLength
        n_mes = 0
        last_measure_offset = part.measures(0, None)[0].offset
        base_measure_offset = part.measures(0, None)[1].offset - part.measures(0, None)[0].offset
        for measure in part.measures(0, None):
            for chord in measure.recurse().addFilter(chord_filter):
                pitches_in_chord = chord.pitches
                for i, pitch in enumerate(pitches_in_chord):
                    pitch = pitch.midi
                    duration = chord.duration.quarterLength
                    velocity = chord.volume.velocity
                    if always_use_measure_offset:
                        offset = int((part.offset + measure_offsets[n_mes] + chord.offset) * sample_freq)
                    else:
                        if abs(measure.offset - last_measure_offset) > base_measure_offset * 2:
                            if measure_offsets:
                                offset = int((part.offset + measure_offsets[n_mes] + chord.offset) * sample_freq)
                            else:
                                return []
                        else:
                            offset = int((part.offset + measure.offset + chord.offset) * sample_freq)
                    note_events.append((pitch, duration, velocity, int(offset)))
            n_mes+=1
            last_measure_offset = measure.offset
    return note_events


def midi_parse_metronome(midi_stream, sample_freq):
    metronome_filter = m21.stream.filters.ClassFilter('MetronomeMark')

    time_events = []
    for part in midi_stream.parts:
        for measure in part.measures(0, None):
            for metro in measure.recurse().addFilter(metronome_filter):
                time = int(metro.number)
                offset = ma.floor((part.offset + measure.offset + metro.offset) * sample_freq)
                time_events.append((time, offset))
    time_events = list(set(time_events))
    time_events = sorted(time_events, key=lambda tup: tup[1])
    return time_events


def midi2notes(midi_stream, sample_freq, transpose_range):
    notes = []
    notes += midi_parse_notes(midi_stream, sample_freq)
    notes += midi_parse_chords(midi_stream, sample_freq)

    # Transpose the notes to all the keys in transpose_range
    return transpose_notes(notes, transpose_range)


def midi2piano_roll(midi_stream, sample_freq, piano_range, transpose_range, stretching_range):
    # Calculate the amount of time steps in the piano roll

    # Parse the midi file into a list of notes (pitch, duration, velocity, offset)
    transpositions = midi2notes(midi_stream, sample_freq, transpose_range)

    time_events = midi_parse_metronome(midi_stream, sample_freq)
    time_streches = strech_time(time_events, stretching_range)
    # 4th index nin tuple is time offset
    # we add 128 just in case
    # remove 128 after processing
    if not transpositions[0]:
        print("Cant process")
        return []
    max_timestep = max(tupl[3] for tupl in transpositions[0]) + 128

    return notes2piano_roll(transpositions, time_streches, max_timestep, piano_range)


def notes2piano_roll(transpositions, time_streches, time_steps, piano_range):
    performances = []

    min_pitch, max_pitch = piano_range
    for t_ix in range(len(transpositions)):
        for s_ix in range(len(time_streches)):
            # Create piano roll with calcualted size.
            # Add one dimension to very entry to store velocity and duration.
            piano_roll = np.zeros((time_steps, MAX_PITCH + 1, 2))

            for note in transpositions[t_ix]:
                pitch, duration, velocity, offset = note
                if duration == 0.0:
                    continue

                # Force notes to be inside the specified piano_range
                pitch = clamp_pitch(pitch, max_pitch, min_pitch)

                piano_roll[offset, pitch][0] = clamp_duration(duration)
                piano_roll[offset, pitch][1] = discretize_value(velocity, bins=40, range=(MIN_VELOCITY, MAX_VELOCITY))

            for time_event in time_streches[s_ix]:
                    time, offset = time_event
                    if offset < piano_roll.shape[0]:
                        piano_roll[offset, -1][0] = discretize_value(time, bins=40, range=(MIN_TEMPO, MAX_TEMPO))

            performances.append(piano_roll)

    return performances


def transpose_notes(notes, transpose_range):
    transpositions = []

    # Modulate the piano_roll for other keys
    first_key = -ma.floor(transpose_range / 2)
    last_key = ma.ceil(transpose_range / 2)

    for key in range(first_key, last_key):
        notes_in_key = []
        for n in notes:
            pitch, duration, velocity, offset = n
            t_pitch = pitch + key
            notes_in_key.append((t_pitch, duration, velocity, offset))
        transpositions.append(notes_in_key)

    return transpositions


def strech_time(time_events, stretching_range):
    streches = []

    # Modulate the piano_roll for other keys
    slower_time = -ma.floor(stretching_range / 2)
    faster_time = ma.ceil(stretching_range / 2)

    # Modulate the piano_roll for other keys
    for t_strech in range(slower_time, faster_time):
        time_events_in_strech = []
        for t_ev in time_events:
            time, offset = t_ev
            s_time = time + 0.05 * t_strech * MAX_TEMPO
            time_events_in_strech.append((s_time, offset))
        streches.append(time_events_in_strech)

    return streches


def discretize_value(val, bins, range):
    min_val, max_val = range

    val = int(max(min_val, val))
    val = int(min(val, max_val))

    bin_size = (max_val / bins)
    return ma.floor(val / bin_size) * bin_size


def clamp_pitch(pitch, max, min):
    while pitch < min:
        pitch += 12
    while pitch >= max:
        pitch -= 12
    return pitch


def clamp_duration(duration, max=THREE_DOTTED_BREVE, min=THREE_DOTTED_32ND):
    # Max duration is 3-dotted breve
    if duration > max:
        duration = max

    # min duration is 3-dotted breve
    if duration < min:
        duration = min

    duration_tuple = m21.duration.durationTupleFromQuarterLength(duration)
    if duration_tuple.type == "inexpressible":
        duration_closest_type = m21.duration.quarterLengthToClosestType(duration)[0]
        duration = m21.duration.typeToDuration[duration_closest_type]

    return duration


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_encoder.py')
    parser.add_argument('--path', type=str, required=True, help="Path to midi data.")
    parser.add_argument('--transp', type=int, default=1, help="Transpose range.")
    parser.add_argument('--strech', type=int, default=1, help="Time stretching range.")
    opt = parser.parse_args()

    # Load data and encoded it
    # multiprocessing_encode(opt.path, n_process=150, transpose_range=opt.transp, stretching_range=opt.strech)
    # load(opt.path, transpose_range=opt.transp, stretching_range=opt.strech)
    # multiprocessing_encode("vgmidi/unlabelled/test/Final_Fantasy_7_LurkingInTheDarkness.mid", n_process=1, transpose_range=opt.transp, stretching_range=opt.strech)
    pitch_range = range(21, 110)
    beat_res ={(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False, 'Bar':False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,
                         'tempo_range': (30, 200)}  # (min, max)
    remi_tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens = True, mask=False)
    # remi_encode("vgmidi/unlabelled/test/Final_Fantasy_7_LurkingInTheDarkness.mid", tokenizer=remi_tokenizer, transpose_range=10)
    remi_multiprocessing_encode(opt.path, tokenizer=remi_tokenizer, transpose_range=opt.transp, n_process=100)
    # print(text)
    #
    # # Write all data to midi file
    # write(text, "encoded.mid")
