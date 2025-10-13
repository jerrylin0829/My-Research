"""
統一日誌系統 - 處理實驗日誌記錄和輸出管理
"""
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any


class ExperimentLogger:
    """實驗日誌管理器"""
    
    def __init__(self, experiment_name: str, output_dir: str, log_level: str = "INFO"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 設置日誌文件路徑
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{experiment_name}_{timestamp}.log")
        
        # 配置日誌系統
        self.setup_logging(log_level)
        
        # 記錄開始
        self.logger.info("="*80)
        self.logger.info(f"{experiment_name} 實驗開始")
        self.logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*80)
    
    def setup_logging(self, log_level: str):
        """設置日誌系統"""
        # 創建logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除既有的handlers
        self.logger.handlers.clear()
        
        # 文件handler - 詳細日誌
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台handler - 簡化輸出
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)  # 只顯示警告和錯誤
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_config(self, config: Dict[str, Any], title: str = "實驗配置"):
        """記錄配置資訊到日誌"""
        self.logger.info(f"\n{title}:")
        self.logger.info("-" * 50)
        
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    self.logger.info(f"{key}: [詳細配置請見config文件]")
                else:
                    self.logger.info(f"{key}: {value}")
        else:
            self.logger.info(str(config))
        
        self.logger.info("-" * 50)
    
    def log_progress(self, current: int, total: int, description: str = "進度"):
        """記錄進度"""
        percentage = (current / total) * 100
        self.logger.info(f"{description}: {current}/{total} ({percentage:.1f}%)")
    
    def log_result(self, result: Dict[str, Any], title: str = "實驗結果"):
        """記錄結果"""
        self.logger.info(f"\n{title}:")
        self.logger.info("-" * 50)
        
        # 記錄關鍵指標
        if 'zrf_score' in result:
            self.logger.info(f"ZRF分數: {result['zrf_score']:.4f}")
        if 'retain_acc' in result:
            self.logger.info(f"保留準確率: {result['retain_acc']:.2f}%")
        if 'forget_acc' in result:
            self.logger.info(f"遺忘準確率: {result['forget_acc']:.2f}%")
        if 'timing' in result:
            self.logger.info(f"執行時間: {result['timing']:.1f}秒")
    
    def log_error(self, error: Exception, context: str = ""):
        """記錄錯誤"""
        error_msg = f"錯誤發生"
        if context:
            error_msg += f" (在 {context})"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg)
        self.logger.debug("詳細錯誤資訊:", exc_info=True)
    
    def save_config_file(self, config: Dict[str, Any], filename: str = "experiment_config.json"):
        """保存配置到JSON文件"""
        config_file = os.path.join(self.output_dir, filename)
        
        # 處理不可序列化的對象
        serializable_config = self._make_serializable(config)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"配置已保存到: {config_file}")
        return config_file
    
    def _make_serializable(self, obj):
        """將對象轉換為可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def finish_experiment(self, success: bool = True):
        """完成實驗記錄"""
        status = "成功完成" if success else "執行失敗"
        self.logger.info("="*80)
        self.logger.info(f"{self.experiment_name} 實驗{status}")
        self.logger.info(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"日誌文件: {self.log_file}")
        self.logger.info("="*80)
        
        return self.log_file


def setup_simple_console_logging():
    """為終端輸出設置簡單的日誌"""
    console_logger = logging.getLogger("console")
    console_logger.setLevel(logging.INFO)
    
    if not console_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        console_logger.addHandler(handler)
    
    return console_logger