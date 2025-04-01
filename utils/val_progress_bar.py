from pytorch_lightning.callbacks import TQDMProgressBar

class SingleLineValidationProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        # 调用父类方法获取默认验证进度条
        val_bar = super().init_validation_tqdm()
        # 设置一个很高的刷新率，确保验证进度条不频繁更新，
        # 从而看起来只在一行上更新（比如，每1000个批次才刷新一次）
        val_bar.refresh_rate = 1000
        # 如果需要，还可以调整描述信息
        val_bar.set_description("Validation")
        return val_bar