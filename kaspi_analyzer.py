from telegram import *
from telegram.ext import *
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import re
import tabula
plt.rcParams["figure.figsize"] = (20,15)
TOKEN = '6162575024:AAHoY_E8aApic93SrrWU9gL8yE62c5VDQ80'

def generate_top_income(df):
    # Create a table from the filtered DataFrame
    filtered_df = df[df['summ'] > 0].groupby('details')['summ'].sum().sort_values(ascending=False).head(50).reset_index()
    filtered_df['summ'] = filtered_df['summ'].apply(lambda x: '{:,}'.format(int(x)).replace(',', ' '))

    # Extract data from the DataFrame
    data = []  # Add column headers
    data.extend(filtered_df.values.tolist())

    # Define column labels
    column_labels = filtered_df.columns.tolist()

    image_width = 1000  # Replace with actual image width
    image_height = 800  # Replace with actual image height

    # fig, ax = plt.subplots(figsize=(image_width * 0.0095, image_height * 0.0095))
    # ax.margins(0)
    fig, ax = plt.subplots()
    # Hide axes
    ax.axis('off')

    # Create a table and add data
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=column_labels)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Set custom subplots_adjust to reduce margins
    # fig.subplots_adjust(left=0.05, right=0.6)

    # Cell Colors, Text Properties, Column Widths
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust cell height
    table.auto_set_column_width([0, 1, 2])  # Adjust column widths
    table.set_fontsize(10)  # Set font size for table content

    # Customize cell colors, text properties, and alignment
    for i, key in enumerate(table.get_celld().keys()):
        cell = table[key]
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor('#666666')
            cell.set_fontsize(12)
            cell.set_text_props(ha='center', va='center')  # Center-align header text
        else:
            cell.set_facecolor('#f3f3f3' if i % 2 == 0 else 'white')
            cell.set_text_props(fontsize=10, ha='center', va='center')  # Center-align content

    # Remove cell borders
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
    # plt.tight_layout()
    plt.tight_layout(pad=0)
    return plt

def generate_top_outcome(df):
    filtered_df = df[df['summ'] < 0].groupby('details')['summ'].sum().sort_values(ascending=True).head(30).reset_index()
    filtered_df['summ'] *= -1
    filtered_df['summ'] = filtered_df['summ'].apply(lambda x: '{:,}'.format(int(x)).replace(',', ' '))

    # Extract data from the DataFrame
    data = []  # Add column headers
    data.extend(filtered_df.values.tolist())

    # Define column labels
    column_labels = filtered_df.columns.tolist()

    # Create a figure and axis
    image_width = 1000  # Replace with actual image width
    image_height = 800  # Replace with actual image height

    # fig, ax = plt.subplots(figsize=(image_width * 0.0095, image_height * 0.0095))
    fig, ax = plt.subplots()
    # Hide axes
    ax.axis('off')

    # Create a table and add data
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=column_labels)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)


    # Cell Colors, Text Properties, Column Widths
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust cell height
    table.auto_set_column_width([0, 1, 2])  # Adjust column widths
    table.set_fontsize(10)  # Set font size for table content

    # Customize cell colors, text properties, and alignment
    for i, key in enumerate(table.get_celld().keys()):
        cell = table[key]
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor('#666666')
            cell.set_fontsize(12)
            cell.set_text_props(ha='center', va='center')  # Center-align header text
        else:
            cell.set_facecolor('#f3f3f3' if i % 2 == 0 else 'white')
            cell.set_text_props(fontsize=10, ha='center', va='center')  # Center-align content

    # Remove cell borders
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)

    return plt

def generate_outcome_by_month(df): 
    df_for_plot_monthly_out = df[df['summ'] < 0].copy()
    df_for_plot_monthly_out = df_for_plot_monthly_out.set_index('date')
    df_for_plot_monthly_out = df_for_plot_monthly_out.resample('1m')['summ'].sum()
    df_for_plot_monthly_out = df_for_plot_monthly_out.reset_index()
    df_for_plot_monthly_out['summ'] *= -1
    sns.lineplot(x = df_for_plot_monthly_out['date'],
                 y = df_for_plot_monthly_out['summ'])
    plt.grid()
    plt.xlabel('Дата', size = 40)
    plt.ylabel('Траты', size = 40)
    plt.xticks(df_for_plot_monthly_out['date'], size = 30, rotation = 45)
    lower_bound = round(df_for_plot_monthly_out['summ'].min(), -3)
    upper_bound = round(df_for_plot_monthly_out['summ'].max(), -3)
    plt.yticks(np.arange(lower_bound,
              upper_bound,
              (upper_bound -
               lower_bound) /10),
               size = 30)
    plt.title('Траты по месяцам', size = 70);
    return plt

def generate_income_by_month(df):
    df_for_plot_monthly_in = df[df['summ'] > 0].copy()
    df_for_plot_monthly_in = df_for_plot_monthly_in.set_index('date')
    df_for_plot_monthly_in = df_for_plot_monthly_in.resample('1m')['summ'].sum()
    df_for_plot_monthly_in = df_for_plot_monthly_in.reset_index()
    sns.lineplot(x = df_for_plot_monthly_in['date'],
                 y = df_for_plot_monthly_in['summ'])
    plt.grid()
    plt.xlabel('Месяц', size = 40)
    plt.ylabel('Доход', size = 40)
    plt.xticks(df_for_plot_monthly_in['date'], size = 30, rotation = 45)
    lower_bound = round(df_for_plot_monthly_in['summ'].min(), -3)
    upper_bound = round(df_for_plot_monthly_in['summ'].max(), -3)
    plt.yticks(np.arange(lower_bound,
              upper_bound,
              (upper_bound -
               lower_bound) /10),
               size = 30)
    plt.title('Доход по месяцам', size = 70);

    return plt


def frquently_outcome(df):
    data_for_common_out = df[df['summ'] < 0].copy()
    data_for_common_out['summ'] *= -1
    text = ''
    for i in data_for_common_out['type'].unique():
        temp = data_for_common_out[data_for_common_out['type'] == i]
        temp = temp.groupby('details')[['summ', 'date']].agg(['count', 'min', 'max', 'mean'])

        # Get the column with the maximum count
        max_count_column = temp['summ', 'count'].idxmax()

        details = max_count_column
        max_count = temp.loc[max_count_column]['summ', 'count']
        min_sum = temp.loc[max_count_column]['summ', 'min']
        max_sum = temp.loc[max_count_column]['summ', 'max']
        mean_sum = round(temp.loc[max_count_column]['summ', 'mean'])
        last_time = temp.loc[max_count_column]['date', 'max']
        first_time = temp.loc[max_count_column]['date', 'min']

        text += f"""Чаще всего вы тратили на {i} {details}.
А именно {max_count}.
Минимальная сумма - {min_sum}.
Максимальная сумма - {max_sum}.
Средняя сумма - {mean_sum}.
Последний раз был - {last_time.date()}.
Первый раз был - {first_time.date()}""" + '\n' + 40*'_' + '\n'
    return text


def frquently_income(df):
    data_for_common_out = df[df['summ'] > 0].copy()
    text = ''
    for i in data_for_common_out['type'].unique():
        temp = data_for_common_out[data_for_common_out['type'] == i]
        temp = temp.groupby('details')[['summ', 'date']].agg(['count', 'min', 'max', 'mean'])

        # Get the column with the maximum count
        max_count_column = temp['summ', 'count'].idxmax()

        details = max_count_column
        max_count = temp.loc[max_count_column]['summ', 'count']
        min_sum = temp.loc[max_count_column]['summ', 'min']
        max_sum = temp.loc[max_count_column]['summ', 'max']
        mean_sum = round(temp.loc[max_count_column]['summ', 'mean'])
        last_time = temp.loc[max_count_column]['date', 'max']
        first_time = temp.loc[max_count_column]['date', 'min']

        text += f"""Чаще всего вы получали от {i} {details}.
    А именно {max_count}.
    Минимальная сумма - {min_sum}.
    Максимальная сумма - {max_sum}.
    Средняя сумма - {mean_sum}.
    Последний раз был - {last_time.date()}.
    Первый раз был - {first_time.date()}""" + '\n' + 40*'_' + '\n'
    return text


def pdf_handler(update, context):
    chat_id = update.effective_user.id
    message = update.message
    if message.document:
        file_id = message.document.file_id
        # file_content = context.bot.get_file(file_id)
        with open('document.pdf', 'wb') as f:
            # f.write(file_content)
            context.bot.get_file(update.message.document).download(out=f)
        tables = tabula.read_pdf('document.pdf', pages='all')
        os.remove('document.pdf')
        df = pd.DataFrame()
        for i in tqdm(range(2,len(tables))):
            temp_df = tables[i].copy()
            a = pd.Series(temp_df.columns).to_frame().T
            temp_df.columns = range(4)
            temp_df = pd.concat([temp_df, a], ignore_index = 1)
            df = pd.concat([df, temp_df], ignore_index = 1)
        df.columns = ['date', 'summ', 'type', 'details']
        df = df[df['summ'] != 'Сумма']
        date_format = "%d.%m.%y"
        df['date'] = df['date'].apply(lambda x: datetime.strptime(x, date_format))
        df['summ'] = df['summ'].apply(lambda x: float(x.split('₸')[0]
                                                          .replace(',', '.')
                                                          .replace(' ', '')))
        top_income_plot = generate_top_income(df)

        img_buffer = io.BytesIO()
        top_income_plot.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        context.bot.send_photo(chat_id=chat_id, photo=img_buffer, caption="Топ трат.")
        top_income_plot.clf()  # Close the figure
        img_buffer.close()  

        top_outcome_plot = generate_top_outcome(df)

        img_buffer = io.BytesIO()
        top_outcome_plot.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        context.bot.send_photo(chat_id=chat_id, photo=img_buffer, caption="Топ доходов.")
        top_outcome_plot.clf()  # Close the figure
        img_buffer.close() 

        outcome_by_month_plot = generate_outcome_by_month(df)

        img_buffer = io.BytesIO()
        top_outcome_plot.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        context.bot.send_photo(chat_id=chat_id, photo=img_buffer, caption="Траты по месяцам.")
        outcome_by_month_plot.clf()  # Close the figure
        img_buffer.close() 

        income_by_month_plot = generate_income_by_month(df)

        img_buffer = io.BytesIO()
        top_outcome_plot.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        context.bot.send_photo(chat_id=chat_id, photo=img_buffer, caption="Траты по месяцам.")
        income_by_month_plot.clf()  # Close the figure
        img_buffer.close() 
        plt.close('all')

        frequently_outcome_text = frquently_outcome(df)
        update.message.reply_text(text=frequently_outcome_text)

        frequently_income_text = frquently_income(df)
        update.message.reply_text(text=frequently_income_text)
    return ConversationHandler.END


updater = Updater(token=TOKEN, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.document, pdf_handler))
updater.start_polling()
print('STARTED')