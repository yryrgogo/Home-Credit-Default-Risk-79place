# home-credit-default-risk
kaggle competition

## Try

### やったこと
前処理（外れ値除去、収縮、欠損値補完・回帰）、エンコーディング一式はSolution見る限りそこまで問題はなかった
特徴作成で試せていなかったのはoof Predictionくらいで他はほぼ作ってた（しかしここで差がついた）
KernelやDiscussionを一通り試した

### やれなかったこと
Posテーブルのクレンジングが十分でなく甘かった。
各テーブル別に予測モデルを作成しoofでStacking（終盤までStackingを使ってCVを伸ばそうというつもりがなかった）
Train/Test分布の違いは知りつつ、その違いをモデルに学習させることはできなかった。
金利特徴作成は甘かった。

### 失敗したこと
ターゲットエンコーディング。リークはしてないが、今回はTrain/Testの分布が異なりLBがアテにならないコンペだったので使うべきでなかった。
→こいつを抜けばPBでもGold圏内に十分入れた
Stackingをよくわかってなかったので後回しにした
終盤までシングルモデルで勝負して多様なモデルを作ろうとしなかった
特徴管理、途中作成したモデルを残してなかった

### 苦労したこと
特徴選択。1つの特徴追加は、よほど強力な特徴でない限り、LGBMのCVで優劣をつけるのが難しい（パラメータや乱数がある）
→CVのfoldで半分以上が改善する特徴を選ぶ
特徴管理。1度作った特徴を消して何度か重複したものを作った気がする。また、10万以上の特徴があったので、どれがどう効くかはメイン以外ほぼ管理できてなかった。
特徴の命名ルール
Ensamble
試行錯誤のログ管理

### TODO
ターゲットエンコーディングでCVとLBが完璧に相関していた理由を確認する（なぜPBだけ伸びなかったのか？）
Sampleに対する予測ウェイトを管理できるようにする
GCP, terminal, pythonの連携周りをスムーズにする。まだ余計ない手間が多い

## Winners Solution
### 3th
SK_ID_PREVレベルで予測を行い、予測値を集計してそのテーブルに関する特徴量とする
モデルはテーブルをマージせず個別に作成し、メインモデルにまとめた
EXTも予測させ、オリジナルとの違いなども特徴とした
INCOMEの予測は失敗した
### 17th
SK_ID_PREVレベルで予測を行い、予測値を集計してそのテーブルに関する特徴量とする
### 7th
時系列feature
異なるデータセットで作った11モデルのstacking, CV0.793のNNなど。
### 16th
普通の集計と時系列featureとGPのstacking。11モデルくらい
### 10th
CNT_PAYMENTを予測するモデルをapplicationの為に作った
金利の特徴をちゃんと作った
cash loanのみのモデル
bureauはActiveテーブルとClosedテーブルを作った。3, 6, 18, 30, 42, 54, 66 monthに切ってそれぞれ特徴を作った。
直近何割のレコード、一番過去の何割かのレコード（0.1~0.4）
SK_ID_CURR, SK_ID_PREVでgroupbyした
oliverのfeature selectionを使った
### 5th
1.
各ID毎に96ヶ月のデータ横持ちの表にして、画像分類としてベクタライズした
CVの各foldに閾値を設定し、その閾値を超えた特徴を選ぶ
2.
isにTARGETをつけ、予測させると、直近でmissing moneyしてる人ほど確率が高くなると出る
like missing money = installment amount - payment amount
これにより、集計や時系列で特徴を作成することなく、その人の振るまいを特徴にできた。そしてこれを複数のメソッドで集計した
要点は、不作用者に共通する少数の行動を特定することです。（isの各行にTARGETをつけて予測することで）
ccbのloan_typeの分布は、trainとtestで比率が大きく異なるので、これを特徴量にしたらCVが大幅に上昇したが、overfitだった
3.
金利はその顧客のリスクに対してつくものなので、金利を知ることは、現在のホームクレジットモデルからリスクアセスメントをある程度知ることを意味します。その瞬間から、列車とテストセットのローン金利を推測する旅を開始しました。
4.
銀行のスコアリングを考え、作った。
appとprevのデータを使い、ロジットモデルとプロビットモデルを作ってoofを特徴量とした
5.
カテゴリのエンコーディング。low~highなどはランク順に数字をつけた
集計はカテゴリ2つの組み合わせをベースにおこなった。
time_windowを変えて特徴を作った

### 2th
Adversarial Stochastic Blending
PostProcessing(duplicate ID Leak)
Try Everything
