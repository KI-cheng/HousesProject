from flask import Flask, render_template, request, jsonify
import mergetest
import test

# 正确设置模板目录和静态文件目录
app = Flask(__name__,
            template_folder='templates',  # 设置模板目录
            static_folder='static',  # 设置静态文件目录
            static_url_path=''
            )


# 定向到index主页
@app.route('/', methods=['GET', 'POST'])
def index():
    sample_size = request.form.get('sample_size')
    if sample_size is not None:
        data_frame = mergetest.ensemble_predict(num_samples=int(sample_size))
        data_frame = data_frame.drop(
            ['mlp_prediction(HKD)', 'rf_prediction(HKD)', 'ensemble_error', 'mlp_error', 'rf_error'], axis=1)
        return render_template('index.html', predict_prices=data_frame)
    else:
        data_frame = mergetest.ensemble_predict()
        data_frame = data_frame.drop(
            ['mlp_prediction(HKD)', 'rf_prediction(HKD)', 'ensemble_error', 'mlp_error', 'rf_error'], axis=1)
    return render_template('index.html', predict_prices=data_frame)


# 这个是给index回显表格用的
@app.route('/query', methods=['GET', 'POST'])
def query():
    sample_size = request.form.get('sample_size')
    print(sample_size)
    # 获取新数据
    new_predict_prices = mergetest.ensemble_predict(num_samples=int(sample_size))
    new_predict_prices = new_predict_prices.drop(
        ['mlp_prediction(HKD)', 'rf_prediction(HKD)', 'ensemble_error', 'mlp_error', 'rf_error'], axis=1)

    # 将DataFrame转换为列表格式
    data = new_predict_prices.values.tolist()

    return jsonify({
        'status': 'success',
        'data': data
    })


@app.route('/charts', methods=['GET', 'POST'])
def chart():
    return render_template('listings.html')


if __name__ == "__main__":
    app.run(debug=True)
