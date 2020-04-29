from flask import Flask, jsonify, request, render_template
import logging as logger
#logger.basicConfig(level="DEBUG")
import aimodule
import sentiment
import requests
import json
app = Flask(__name__,template_folder = './templates')

@app.route('/')
def hello():
    aimodule.train()
    #return {"message" : "Inside hello method"},200
    return render_template('chat.html')
    
@app.route('/ask', methods=['POST'])
def ask():
    # kernel now ready for use
    while True:
        message = str(request.form['messageText'])
        m1 = message[:5]
        if message == "quit":
            aimodule.record()
        elif message == "save":
            aimodule.saveBrain("bot_brain.brn")
        elif message == "#elp":
            aimodule.incident()
        else:
            bot_response = aimodule.respond(message)
            sen_response = sentiment.predict(message)
            resp = bot_response + "---" +sen_response
            #print sen_response
            logger.debug(sen_response)
            return jsonify({'status':'OK','answer':resp})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug="False",use_reloader="True")
