from enum import Enum
class book_features_values:
    def __init__(self):
        self.WritingStyles = {u"童话":1, u"寓言":2, u"故事":3, u"儿童小说":4, u"人物传记类":5, u"诗歌":6, u"散文":7, u"小说":8, u"话剧类（多幕剧、独幕剧等）":9}
        self.Theme = {u"自然动物类":1, u"亲情友情类":2, u"校园生活类":3, u"成长类（梦想、坚强、感恩、人物传记等）":4, u"生命生存类（乡村、城市等）":5, u"社会政治类（环保、种族、战争等）":6, u"历史类":7, u"武侠类":8, u"文化哲学类":9}
        self.TopicDepth = {u"浅显":1, u"适中":2, u"深奥":3}
        self.LanguageStyle = {u"作家地域、民族、时代、流派":1, u"作品平实质朴，朴素自然":2, u"作品含蓄隽永，文学韵味浓厚":3, u"作品想象奇特，充满浪漫气息":4, u"作品幽默风趣，富有讽刺、哲理意味":5, u"作品形象、生动":6}
        self.CharacterNum = {u"1—5万字":1, u"5—8万字":2, u"8—10万字":3, u"10—15万字":4,u"15—20万字":5, u"20万字以上":6}
        self.ImageRatio = {u"配图丰富，图文并茂":1, u"有一定的配图":2, u"配图较少":3, u"几乎无配图":4}
        self.LabelGrade = [u"1,2", u"3,4", u"5,6", u"7", u"8,9", u"11,12"]
    def get_feature_values(self,feature_name):
         if book_features.WritingStyles.name == feature_name:
             return self.WritingStyles
         if book_features.Theme.name == feature_name:
             return self.Theme
         if book_features.TopicDepth.name == feature_name:
             return self.TopicDepth
         if book_features.LanguageStyle.name == feature_name:
             return self.LanguageStyle
         if book_features.CharacterNum.name == feature_name:
             return self.CharacterNum
         if book_features.ImageRatio.name == feature_name:
             return self.ImageRatio
         if book_features.LabelGrade.name == feature_name:
             return self.LabelGrade

    def get_features_name(self,feature_index):
       if book_features.WritingStyles.value[0] == feature_index:
           return book_features.WritingStyles.name
       if book_features.Theme.value[0] == feature_index:
           return book_features.Theme.name
       if book_features.TopicDepth.value[0] == feature_index:
           return book_features.TopicDepth.name
       if book_features.LanguageStyle.value[0] == feature_index:
           return book_features.LanguageStyle.name
       if book_features.CharacterNum.value[0] == feature_index:
           return book_features.CharacterNum.name
       if book_features.ImageRatio.value[0] == feature_index:
           return book_features.ImageRatio.name
       if book_features.LabelGrade.value[0] == feature_index:
           return book_features.LabelGrade.name
# 每个年级浮动范围
GradeThreshold = {1: 0.75, 2: 0.75, 3: 0.75, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1.25, 10: 1.25, 11: 1.25, 12: 1.25}
# cols = ["书名","体裁", "题材", "主题深度", "语言风格", "字数", "图片比例", "年级"]
Cols=['BookName','WritingStyles','Theme','TopicDepth','LanguageStyle','CharacterNum','ImageRatio','LabelGrade']
class book_features(Enum):
    WritingStyles= 0,#"体裁"
    Theme = 1 ,#"题材"
    TopicDepth = 2,#"主题深度"
    LanguageStyle = 3, #"语言风格"
    CharacterNum = 4,#"字数"
    ImageRatio = 5,#"图片比例"
    LabelGrade = 6,#"年级"