"""
run_crawler.py — Chạy crawler UIAutomation.
"""
import logging
import time
import yaml

from src.crawler import ScreenCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # crawler = ScreenCrawler(
    #     output_dir=config["paths"]["real_data"],
    #     max_samples=config["crawler"]["max_samples"],
    # )
    # thread = crawler.run_background(interval_sec=config["crawler"]["interval_sec"])
    
    # try:
    #     while thread.is_alive():
    #         time.sleep(5)
    #         print(f"  Samples: {crawler.count:,}", end="\r")
    # except KeyboardInterrupt:
    #     crawler.stop()
    #     thread.join(timeout=5)
    #     print(f"\nDừng — đã thu: {crawler.count:,} samples")
    
    print("Đang phát triển: crawler chưa hoàn thiện")
