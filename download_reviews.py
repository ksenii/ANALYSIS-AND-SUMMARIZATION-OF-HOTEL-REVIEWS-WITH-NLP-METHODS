"""
Collecting hotel reviews from Yandex.Travel
Output data: date, review, rating, authorization, likes, dislikes
Output saved in csv-file
"""

import csv
import time
import logging
import random
import re
import os
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('yandex_review_collecting.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Dictionary for Russian month conversion
MONTHS = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12}


def clear_output_file(output_csv, keep_backup=True):
    """Clears output file with optional backup"""
    try:
        if not os.path.exists(output_csv):
            logger.warning(f"Output file not found: {output_csv}")
            return True
        if keep_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{output_csv}.backup_{timestamp}"
            os.rename(output_csv, backup_name)
            logger.info(f"Created backup: {backup_name}")
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["date", "review", "rating", "authorization", "likes", "dislikes"])
            writer.writeheader()
        logger.debug("Output file cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to clear output file: {str(e)}", exc_info=True)
        return False


def load_existing_reviews(output_csv):
    """Loads existing reviews from csv file"""
    try:
        if not os.path.exists(output_csv):
            logger.debug("No existing reviews file found")
            return []
        with open(output_csv, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_reviews = [row["review"] for row in reader]
            logger.info(f"Loaded {len(existing_reviews)} existing reviews")
            return existing_reviews
    except Exception as e:
        logger.error(f"Failed to load existing reviews: {str(e)}", exc_info=True)
        return []


def parse_russian_date(date_str):
    """Transforms string date to datetime"""
    try:
        if not date_str or not isinstance(date_str, str):
            logger.debug("Invalid date string")
            return None
        date_str = re.sub(r'\s+', ' ', date_str.strip().lower())
        logger.debug(f"Processing date string: {date_str}")
        if date_str == 'сегодня':
            return datetime.now()
        elif date_str == 'вчера':
            return datetime.now() - timedelta(days=1)
        date_str = date_str.replace('.', '')
        parts = [p for p in date_str.split(' ') if p]
        if len(parts) == 3:  # Full date with year
            day, month, year = parts
            year = int(year)
        elif len(parts) == 2:  # Date without year
            day, month = parts
            current_date = datetime.now()
            year = current_date.year
            test_date = datetime(year, MONTHS[month], int(day))
            if test_date > current_date:
                year -= 1
        else:
            logger.warning(f"Error date format: {date_str}")
            return None
        month_num = MONTHS.get(month)
        if not month_num:
            logger.warning(f"Error month: {month}")
            return None
        parsed_date = datetime(year, month_num, int(day))
        logger.debug(f"Successfully processed date: {parsed_date}")
        return parsed_date
    except Exception as e:
        logger.error(f"Date processed failed: {str(e)}", exc_info=True)
        return None


def is_recent_review(review_date, months_threshold=6):
    """Checks if review is recent"""
    try:
        if not review_date:
            logger.debug("No review date provided")
            return True
        if isinstance(review_date, str):
            review_date = datetime.strptime(review_date, "%Y-%m-%d")
        threshold_date = datetime.now() - timedelta(days=30 * months_threshold)
        is_recent = review_date >= threshold_date
        return is_recent
    except Exception as e:
        logger.error(f"Date filtering failed: {str(e)}", exc_info=True)
        return True


def save_reviews(reviews_data, output_csv, clear_file=False):
    """Saves reviews to csv"""
    try:
        if not reviews_data:
            logger.warning("No reviews saved")
            return
        logger.info(f"Saving {len(reviews_data)} reviews to {output_csv}")
        write_header = clear_file or not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
        mode = 'a' if not clear_file else 'w'
        with open(output_csv, mode, encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["date", "review", "rating", "authorization", "likes", "dislikes"])
            if write_header:
                writer.writeheader()
            writer.writerows(reviews_data)
        logger.info("Reviews saved successfully")
    except Exception as e:
        logger.error(f"Failed save reviews: {str(e)}", exc_info=True)


def parse_review_block(block):
    """Review block processing"""
    try:
        logger.debug("Starting review block processing")
        # Extract review text
        review_text = block.find("div", class_="RYSdb").find("div", class_="lpglK Eqn7e b9-76").text.strip()
        # Extract date
        parsed_date = None
        date_div = block.find("div", class_="l9Dh5 BUTjn b9-76")
        if date_div:
            date_span = date_div.find("span", class_="Eqn7e dNANh")
            if date_span:
                raw_date_text = date_span.text.strip()
                parsed_date = parse_russian_date(raw_date_text)
        # Extract rating
        rating_elements = block.find_all("div", class_="LFBXk KN22d")
        rating = sum(1 for el in rating_elements if el.get('aria-selected') == 'true')
        # Extract authorization
        auth = False
        try:
            container = block.find("div", class_="EhCXF root fOWDr VPvYa zDZzf _6aDIU wU8vz iF9-T")
            if container:
                auth_span = container.find("span", class_="meqL2 LxKIN ipwvr fEl7Z")
                if auth_span:
                    auth = True
        except Exception as e:
            logger.debug(f"Authorization check failed: {str(e)}")
        # Extract likes/dislikes
        likes = dislikes = 0
        try:
            likes_span = block.select_one("button.WvMZr span.llji2 span._44BiE span.eh5Br span.ETcmP")
            if likes_span and likes_span.text.strip():
                likes = int(likes_span.text.strip())
            dislikes_span = block.select_one("button.WvMZr.Bnnv4.-ZlQV.jNuum.button_width_auto.rrVlD.linkButton_hover:nth-of-type(2) span.eh5Br span.ETcmP")
            if dislikes_span and dislikes_span.text.strip():
                dislikes = int(dislikes_span.text.strip())
        except Exception as e:
            logger.debug(f"Failed to parse reactions: {str(e)}")
        result = {"date": parsed_date.strftime("%Y-%m-%d") if parsed_date else None,
            "review": review_text,
            "rating": rating,
            "authorization": auth,
            "likes": likes,
            "dislikes": dislikes}
        logger.debug(f"Successfully extracted review: {result}")
        return result
    except Exception as e:
        logger.error(f"Review extraction failed: {str(e)}", exc_info=True)
        return None


def apply_fresh_sorting(driver):
    """Applies newest-first sorting"""
    try:
        logger.info("Attempting to use newest-first sorting")
        sort_button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.LMibf label.B1gow.LfP2v")))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sort_button)
        time.sleep(random.uniform(0.5, 1.5))
        sort_button.click()
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "div.qakch")))
        sort_options = driver.find_elements(By.CSS_SELECTOR, "div.qakch")
        target_option = next((opt for opt in sort_options if "Сначала свежие" in opt.text), None)
        if not target_option:
            logger.warning("Newest-first option not found")
            return False
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_option)
        time.sleep(random.uniform(0.5, 1.5))
        target_option.click()
        time.sleep(random.uniform(2, 4))
        logger.info("Newest-first sorting used successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to use sorting: {str(e)}", exc_info=True)
        return False


def extract_reviews(url, output_csv="hotel_reviews.csv", max_months_old=6, clear_existing=True, status_callback=None):
    """Main extract function"""
    def log_status(message, level=logging.INFO):
        if status_callback:
            status_callback(message)
        logger.log(level, message)
    log_status(f"Starting review extraction for URL: {url}")
    log_status(f"Output file: {output_csv}")
    try:
        # Initialize WebDriver
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        driver = webdriver.Chrome(options=options)
        log_status("WebDriver initialized successfully")
        try:
            if clear_existing:
                if clear_output_file(output_csv):
                    log_status("Output file cleared successfully")
                else:
                    log_status("Failed to clear output file", logging.WARNING)
            log_status(f"Loading URL: {url}")
            driver.get(url)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.qEMqC")))
            log_status("Page loaded successfully")
            # Apply sorting
            if not apply_fresh_sorting(driver):
                log_status("Proceeding without newest-first sorting", logging.WARNING)
            # Load existing reviews
            existing_reviews = load_existing_reviews(output_csv)
            reviews_data = []
            unique_reviews = set(existing_reviews)
            attempts = 0
            max_attempts = 5
            no_more_reviews = False
            while attempts < max_attempts and not no_more_reviews:
                current_reviews = driver.find_elements(By.CSS_SELECTOR, "section.root.esfDh.xa7LR")
                log_status(f"Found {len(current_reviews)} reviews on current page", logging.DEBUG)
                new_reviews_count = 0
                stop_processing = False
                # Process each review
                for i, review in enumerate(current_reviews, 1):
                    try:
                        parsed = parse_review_block(BeautifulSoup(review.get_attribute('outerHTML'), 'html.parser'))
                        if not parsed:
                            continue
                        log_status(f"Processing review {i}/{len(current_reviews)}", logging.DEBUG)
                        if parsed['review'] in unique_reviews:
                            log_status("Skipping duplicate", logging.DEBUG)
                            continue
                        if parsed['date'] and not is_recent_review(parsed['date'], max_months_old):
                            log_status(f"Found old review, stopping collection")
                            stop_processing = True
                            break
                        reviews_data.append(parsed)
                        unique_reviews.add(parsed['review'])
                        new_reviews_count += 1
                    except Exception as e:
                        log_status(f"Error processing review {i}: {str(e)}", logging.ERROR)
                if stop_processing:
                    break
                log_status(f"Processed {new_reviews_count} new reviews")
                # Save collected reviews
                if reviews_data:
                    save_reviews(reviews_data, output_csv, clear_existing and (attempts == 0))
                    reviews_data = []
                    clear_existing = False
                # Load more reviews
                try:
                    load_more_buttons = driver.find_elements(By.CSS_SELECTOR, "div.ZfjL0 button[type='button']")
                    if not load_more_buttons:
                        log_status("No more reviews button found", logging.INFO)
                        no_more_reviews = True
                        break
                    button = WebDriverWait(driver, 15).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "div.ZfjL0 button[type='button']")))
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                    time.sleep(random.uniform(1, 2))
                    driver.execute_script("arguments[0].click();", button)
                    try:
                        WebDriverWait(driver, 15).until(
                            lambda d: len(d.find_elements(By.CSS_SELECTOR, "section.root.esfDh.xa7LR")) > len(current_reviews))
                        attempts = 0
                        log_status("Loaded additional reviews successfully")
                    except Exception as e:
                        log_status(f"No new reviews loaded: {str(e)}", logging.WARNING)
                        no_more_reviews = True
                        break
                    time.sleep(random.uniform(3, 5))
                except Exception as e:
                    attempts += 1
                    log_status(f"Loading attempt {attempts}/{max_attempts} failed: {str(e)}", logging.WARNING)
                    time.sleep(random.uniform(2, 4))
            log_status(f"Total reviews collected: {len(unique_reviews)}")
        except Exception as e:
            log_status(f"Collection failed: {str(e)}", logging.CRITICAL)
            raise
        finally:
            driver.quit()
            log_status("WebDriver closed")
    except Exception as e:
        log_status(f"Error: {str(e)}", logging.CRITICAL)
        raise
