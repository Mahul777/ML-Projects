from flask import Flask, request, render_template
from predict_pipeline import CustomData, PredictPipeline

# ✅ Rename app to application (this is needed for AWS Beanstalk)
application = Flask(__name__)

# 📌 Home route
@application.route('/')
def home_page():
    return render_template('home.html')

# 📌 Prediction route
@application.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # 🔄 Get data from form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # 🧪 Get the data in DataFrame format
        final_new_data = data.get_data_as_dataframe()

        # 🔮 Make prediction
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        # 🔁 Show result in template
        return render_template('home.html', results=round(pred[0], 2))

    return render_template('home.html')

# ✅ Only run the app locally (not needed for Elastic Beanstalk)
if __name__ == '__main__':
    application.run(debug=True)
