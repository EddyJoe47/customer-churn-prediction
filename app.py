from flask import Flask, render_template, request
import pandas as pd
import os
import plotly
import plotly.express as px
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    table_html = None
    stats_html = None
    graphJSON = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in request."
        file = request.files['file']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Read CSV
            df = pd.read_csv(filepath)

            # Display dataset
            table_html = df.to_html(classes='table table-striped', index=False)

            # Summary statistics
            stats_html = df.describe(include='all').to_html(classes='table table-bordered')

            # Create Churn bar chart (robust version)
            if 'Churn' in df.columns:
                counts = df['Churn'].value_counts()
                churn_counts = pd.DataFrame({
                    'Churn': counts.index,
                    'Count': counts.values
                })

                fig = px.bar(
                    churn_counts,
                    x='Churn',
                    y='Count',
                    labels={'Churn': 'Churn', 'Count': 'Count'},
                    title='Churn Distribution'
                )
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', table=table_html, stats=stats_html, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
