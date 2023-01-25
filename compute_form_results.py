import json

import pandas as pd
import gspread

SPREADHSHEET_NAME = "MusicGenerationFormDB"
FORM_SHEET_NAME = "Form"
ANSWER_SHEET_NAME = "Answer"


def fetch_data():
    gc = gspread.service_account()
    spreadsheet = gc.open(SPREADHSHEET_NAME)
    answer_sheet = spreadsheet.worksheet(ANSWER_SHEET_NAME)
    answer_data = pd.DataFrame(answer_sheet.get_all_records())
    form_sheet = spreadsheet.worksheet(FORM_SHEET_NAME)
    form_data = pd.DataFrame(form_sheet.get_all_records())
    with open('midi_json', 'r') as data_file:
        json_data = data_file.read()
    pieces_data = pd.DataFrame(json.loads(json_data))

    return answer_data, form_data, pieces_data

def join_all_tables(answer_data, form_data, pieces_data):
    joined_data = pd.merge(answer_data, form_data[['Form ID', 'Music Knowledge', 'Game Knowledge']], on="Form ID")
    joined_data = pd.merge(joined_data, pieces_data[['Piece ID', 'sentiment', 'algorithm']], on="Piece ID")
    return joined_data

def get_pieces_stats(joined_data, pieces_data):
    # groupby and loop for pieces_stats.csv
    result_rows = []
    for piece_name, piece_answers in joined_data.groupby('Piece ID'):
        mean_quality = round(float(piece_answers['Quality'].mean()), 2)
        mean_sentiment = round(float(piece_answers['Sentiment'].mean()), 2)
        sum_turing = int(piece_answers['Turing'].sum())
        result_record = pd.DataFrame(
            [{'Piece ID': piece_name, 'Quality': mean_quality, 'Sentiment': mean_sentiment, 'Turing': sum_turing}])
        concat = pd.merge(pieces_data, result_record, on="Piece ID")
        result_rows.append(concat)
    pieces_stats = pd.concat(result_rows, ignore_index=True)
    pieces_stats.sort_values(by=['algorithm', 'Quality'])
    pieces_stats.to_csv('pieces_results.csv', index=False)
    return pieces_stats

def get_algorithm_stats(joined_data):

    algorithm_data = joined_data.groupby(['algorithm', 'sentiment'])
    algorithm_results = []
    for group_data, group_answers in algorithm_data:
        algorithm_name, sentiment = group_data
        mean_quality = round(float(group_answers['Quality'].mean()), 2)
        mean_quality_std = round(float(group_answers['Quality'].std()), 2)
        mean_sentiment = round(float(group_answers['Sentiment'].mean()), 2)
        mean_sentiment_std = round(float(group_answers['Sentiment'].std()), 2)
        mean_turing = round(int(group_answers['Turing'].sum()) / len(group_answers), 2)
        result_record = pd.DataFrame([{'algorithm': algorithm_name, 'sentiment': sentiment, 'Quality': mean_quality,
                                       'Quality_STD': mean_quality_std,
                                       'Sentiment': mean_sentiment, 'Sentiment_STD': mean_sentiment_std,
                                       'Turing': mean_turing}])
        algorithm_results.append(result_record)
    final_algorithm_results = pd.concat(algorithm_results, ignore_index=True)
    final_algorithm_results.to_csv('algorithm_results.csv', index=False)
    return final_algorithm_results

def get_algorithm_knowledge_stats(joined_data):
    # groupby and loop for algorithm_knowledge_stats.csv
    algorithm_results = []
    grouped_answer_data = joined_data.groupby(['algorithm', 'sentiment','Game Knowledge','Music Knowledge'])
    for group_data in grouped_answer_data:
        algorithm_name, sentiment, group_game_knowledge, group_music_knowledge = group_data[0]
        group_answers = group_data[1]
        mean_quality = round(float(group_answers['Quality'].mean()), 2)
        mean_quality_std = round(float(group_answers['Quality'].std()), 2)
        mean_sentiment = round(float(group_answers['Sentiment'].mean()), 2)
        mean_sentiment_std = round(float(group_answers['Sentiment'].std()), 2)
        mean_turing = round(int(group_answers['Turing'].sum()) / len(group_answers), 2)
        result_record = pd.DataFrame([{'algorithm': algorithm_name, 'sentiment': sentiment, 'Game Knowledge':group_game_knowledge,
                                       'Music Knowledge':group_music_knowledge,'Quality': mean_quality,
                                       'Quality_STD': mean_quality_std,
                                       'Sentiment': mean_sentiment, 'Sentiment_STD': mean_sentiment_std,
                                       'Turing': mean_turing}])
        algorithm_results.append(result_record)
    final_algorithm_results = pd.concat(algorithm_results, ignore_index=True)
    final_algorithm_results.to_csv('algorithm_knowledge_results.csv', index=False)
    return final_algorithm_results

if __name__ == "__main__":
    answer_data, form_data, pieces_data = fetch_data()
    # filter certain user if desired
    # answer_data = answer_data[answer_data['Form ID']=='f2f7e6da-2551-4ca0-b91e-c044f83ac7bd']
    joined_data = join_all_tables(answer_data, form_data, pieces_data)

    pieces_stats = get_pieces_stats(joined_data, pieces_data)
    algorithm_stats = get_algorithm_stats(joined_data)
    algorithm_knowledge_stats = get_algorithm_knowledge_stats(joined_data)
