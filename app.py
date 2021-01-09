# Load Packages

from flask import Flask, render_template, url_for, request
import random as rn
#import textblob
#from textblob import TextBlob
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Create our Flask Object Instance

app = Flask(__name__)

# Set Home Page Template

@app.route('/')
def home():
    sentances = ["اللهم أعننا علي ذكرك وشكرك وحسن عبادتك",
                 " طب ايش هو الخلاف بالتحديد حتي يصير كل هالعقوبه ",
                 "ساعتين جد وكوباية شاي والمنهج ده هجيبه في شوال يابني انت والله متقلقش",
                 "هي المشرحة ناقصه قتله  فكك مننا وروح شوف بلد تانيه تصيع فيها يابني انت ",
                 "وي يا علي وي وي يا علي ويي وي وي وي القاضييييه ممكن علييي معلولل  لا لا هذا ولدنا افشه محمد مجدي قفشه",
                 "الاهلي خد بطولة كاس مصر وخد الدوري كمان عليه وانتو خليكو في المهلبيه اللي انتو فيها",
                 "انت لسه مبقتش من شركاء اوبر !! :/الفرصه لسه فأديك .. و بـ 4 خطوات بس ;)1- هتجيب فيش وتشبيه موجه لشركه",
                 "لو نمت دقيقة كمان زيادة هديك علي دماغك",
                 " مادري شفيني عليه  ",
                 " ياخي شلون تبي تحقق مونديال وانت بدونه ضايع ",
                 " واحد شاورما ختي بدي كتير تومية ويسلمو عيونك ",
                 "   ياليت والله علشان  يريحنا",
                 "يا راجل فن ايه هو ده فن دي قلة ادب",
                 "انا مش عارف انا ايه جابني هنا",
                 " ياخي شلون تبي تحقق مونديال وانت بدونه ضايع",
                 " مادري شفيني عليه ",
                 " ليه بتقول كده علينا طيب يعم انت مالك اصلا؟",
                 "شنو اللي مضايقج مني حبيبي اذ ماتكلمت راح نخسر بعضنا هيك بهالطريقة ؟",
                 "لااااءء وربي مو كشخه ابببددداً يارب انه يمزح",
                 "والله مدري كذا فجأه والشيله اسمها اهجد اهجد",
                 "لو تدور ما تلاقي منهو غيري يموت فييك",
                 "شنو سالفه البث  شو الفيس متروس بس بث",
                 "ياجماعة ودي أعبر لكم عن شوفي بس النت مايساعد!!!!!حتى التويت ذا مدري وشلون نزل",
                 "لو بيسكّر بوزروو بقى لعمى بقلبوا ما ضجر وما فهم انّوا ما حدا بالبّلد تايقوا لعمى شو انّو حمار بس كل عمرو هيك ريكوو",
                 "كل شوي يسألوني شو تغير في وشو صرلي عنن يختفي",
                 " مش قادره اكل اشي بتعرفي هاد الشعور الي تشوفي اكل او بدك تاكلي بتصيري بدك تستفرغي ! الـGaging ؟",
                 "تورطت وصرت عربي العمى على سنة هالحياة شو بتخزي",
                 "نظراته الحزينة مو طبيعي شو معبره 😢😢😢",
                 "تؤبرني توني ؟! الحمدالله انك مو لبناني",
                 "ولا أنا ما قدرت يعليهن 😥 الله يستر  بلكي بلحق محاضرة الساعة12 ولا شو",
                 "فيروز   تذكرتك ياعلياء  وتذكرت عيونك  يخرب بيت عيونك ياعلياء شو حلوين",
                 "قول إنى منيح مايجراش حاجه",
                 "معيش مصاري ولا سيارة وخطي مفصول ومكانش في داعي اداوم بس اجيت لإني غبية وهلا بدي اروح وفش حدا يروحني ولازم ادرس للإمتحان وبس. صباح الخير🤦🏽‍♀️",
                 " وحدة بتسأل حبيبها شو بدك تهديني على عيد الفلنتاين ! 🙈🙈 حكالها بدي اهديكي عالصراط المستقيم ونتوب ونترك بعض .🐸  تكبيررررررر…",
                 "أنا بدي ياك و أنت منك عارف شو بدك",
                 "اما.بالنسبه لغاليتي هي يلي بتضل احلا شي عرفتو هون بعمري ما شفت حدا رقيق وحساس متلا",
                 "ضربتين بالرأس بوجعن  لا مصاري ولا دخان والله مشكله",
                 " البنات : شو الامتحان ؟ زفت  49 من 50 الشباب : شو الامتحان ؟ تفجير 5 من 50  لا إحساس ولا ضمير واخر شي هو ضابط وهي تبيع فطاير…",
                 "✋مخنوك😔  #ربــي اخذنــي الدنيــا ماترتاحلــي😖   كلش ضوجه",
                 "أنا مع الجهّال جاهل بـ كلش وأنا مع الصاحين صاحي وجاهز من وين ما تبغى نبلش نبلش خذني على كل الفرص ذيب ناهز",
                 "ماشوف احد يركض وراي 🏃🏽🏃🏽🏃🏽 ترى كلش مافيني لياقة",
                 " جيتك غلا ما جيت لك اترجاك يومك تكلمني من راس خشمك  جيتك مقدر عشرتن كنت وياك لا عاده الله ذا   الغلا فز يمك   #تايه",
                 "تدري ليش ماابي اصير حلوه عشان ماتقولون عني هالكلبه حلوه هالحماره حلوه يخي كلش ولااحد يغلط علي 💅",
                 "انا اللي هموت واعرف جبت الارقام دي منين ؟ ياريت ماتحرمناش من مصادرك",
                 "مش متخيله انه ممكن ميد بتاع بكرا كان هيروح عليا وانا قاعده نايمه متأنتخه ف السكن 😑ومعرفش بردوا غير لما يضيع",
                 "ربنا بيبعتلك علامات واضحه جدا اوقات وانت بتصمم تكمل علشان غبى مفيش اى امل ان الواحد يفتح كتاب بكرا 😅",
                 "😂😂😂😂😂 انا حفظت شويه امبارح  بس مفيش حد بشجعني يالولي اعمل ايه زهههقت",
                 "احنا هنعمل كل حاجة تحسسك انك زيك زي اي حد في حياتنا بس هنيجي في اخر اليوم وهنقولك انك مش زي أي حد وانك أهم منهم كلهم",
                 "انا مش عارفه هي الناس تعاملنا كويس وقت مايكون مزاجها حلو وتعاملنا وحش لما يكون مزاجها وحش انتو ايه عبط يعني ولا ايه ؟"


                 ]
    def get_ran():
        return (rn.choice(sentances))

    return render_template('home.html',sentances =sentances , rn =rn , get_ran = get_ran())

    # Set Predict Page
# Using TextBlob package (powered by the Google Translate API)

@app.route('/predict',methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        data2 = pd.read_csv('Arabic_Data_cleaned_without_duplicated.csv')
        text2 = data2['Arabic_Tweets_Cleaned'].values.tolist()
        target2 = data2['labels_new'].values.tolist()
        X_train2, X_test2, y_train2, y_test2 = train_test_split(text2, target2, test_size=0.2, shuffle=True,random_state=42)
        X_train_counts2 = count_vect.fit_transform(X_train2)
        X_tfidf_train2 = tfidf_transformer.fit_transform(X_train_counts2)
        #X_test_counts2 = count_vect.transform(X_test2)
        #X_tfidf_test2 = tfidf_transformer.transform(X_test_counts2)
        clf_NB = MultinomialNB().fit(X_tfidf_train2, y_train2)
#######################################################################################################3
        #svm_clf = svm.SVC()
        #svm_clf = svm_clf.fit(X_tfidf_train2, y_train2)

        #joblib.dump(svm_clf, 'svm_Arabic_model2.pkl')
        svm_arabic_model = open('svm_Arabic_model2.pkl', 'rb')
        clf_svm = joblib.load(svm_arabic_model)

        #####################################################################################################
        message = request.form['message']
       # message = str(message)
        #print(str(message))
        message2 = message
        message = [message]
  ##########################################################################################################
        X_testing_counts = count_vect.transform(message)
        X_tfidf_testing = tfidf_transformer.transform(X_testing_counts)
        detect = clf_NB.predict(X_tfidf_testing)
        print(detect)
        ##########################################################################################
        #########################################################################################
        #joblib.dump(model, 'XG_Arabic_model2.pkl')
        XG_arabic_model = open('XG_Arabic_model2.pkl', 'rb')
        clf_xg = joblib.load(XG_arabic_model)
        detect2 = clf_xg.predict(X_tfidf_testing)
        print(detect2)


        detect3 = clf_svm.predict(X_tfidf_testing)
        print(detect3)

        #blobline = TextBlob(message)
        #detect = blobline.detect_language()
    return render_template('result.html',prediction = detect,prediction2 = detect2,prediction3 = detect3,messages = message2)

if __name__ == '__main__':
	app.run()