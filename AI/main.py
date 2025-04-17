from author_encoder import MultiExpertEncoder
from user_encoder import ContentEncoder
from generator import StyleFusion, CalligraphyGenerator

class CalligraphySystem:
    def __init__(self, author_model_path):
        # 初始化各模块
        self.author_encoder = MultiExpertEncoder.load(author_model_path)
        self.user_encoder = ContentEncoder()
        self.fusion_layer = StyleFusion()
        self.generator = CalligraphyGenerator()
        
    def process(self, user_img, author_style_id):
        # 特征提取
        author_style = self.author_encoder.get_style(author_style_id)  # [1,512]
        user_feat = self.user_encoder(user_img)  # {'visual': ..., 'structure': ...}
        
        # 特征融合
        fused_feat = self.fusion_layer(
            author_style, 
            user_feat['structure']
        )
        
        # 书法生成
        output_img = self.generator(fused_feat)
        return output_img