"""
Automate the DataMigrationCrane flow and capture screenshots.
Run: pip install playwright && playwright install chromium
Then: python screenshot_flow.py
"""
import asyncio
import time
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Install: pip install playwright && playwright install chromium")
    raise

OUTPUT_DIR = Path(__file__).parent / "screenshots"
BASE_URL = "http://127.0.0.1:5000"


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Screenshots will be saved to: {OUTPUT_DIR}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        try:
            # 1. Source Ingestion page
            print("1. Navigating to Source Ingestion...")
            await page.goto(BASE_URL, wait_until="networkidle")
            await page.wait_for_timeout(1500)  # Wait for example to load
            await page.screenshot(path=OUTPUT_DIR / "01_source_ingestion.png", full_page=True)
            print("   Saved: 01_source_ingestion.png")

            # 2. Click Submit & Discover Joins
            print("2. Clicking 'Submit & Discover Joins'...")
            submit_btn = page.locator('button:has-text("Submit")').first
            await submit_btn.click()
            await page.wait_for_timeout(3000)  # Wait for API call
            await page.wait_for_url("**/source/visualize**", timeout=10000)
            await page.wait_for_timeout(2000)  # Wait for graph to render

            # 3. Source Schema Visualization
            print("3. Capturing Source Schema Visualization...")
            await page.screenshot(path=OUTPUT_DIR / "03_source_visualization.png", full_page=True)
            print("   Saved: 03_source_visualization.png")

            # 4. Click Continue to Target Ingestion
            print("4. Clicking 'Continue to Target Ingestion'...")
            await page.click('a:has-text("Continue to Target Ingestion")')
            await page.wait_for_url("**/target/ingest**", timeout=5000)
            await page.wait_for_timeout(1500)

            # 5. Target Ingestion page
            print("5. Capturing Target Ingestion...")
            await page.screenshot(path=OUTPUT_DIR / "05_target_ingestion.png", full_page=True)
            print("   Saved: 05_target_ingestion.png")

            # 6. Click Map Source to Target
            print("6. Clicking 'Map Source to Target'...")
            map_btn = page.locator('button:has-text("Map Source to Target")').first
            await map_btn.click()
            await page.wait_for_timeout(8000)  # Mapping can take several seconds
            await page.wait_for_url("**/target/visualize**", timeout=20000)
            await page.wait_for_timeout(2000)

            # 7. Target Mapped Visualization
            print("7. Capturing Target Mapped Visualization...")
            await page.screenshot(path=OUTPUT_DIR / "07_target_visualization.png", full_page=True)
            print("   Saved: 07_target_visualization.png")

            print("\nAll screenshots captured successfully!")

        except Exception as e:
            print(f"\nError: {e}")
            await page.screenshot(path=OUTPUT_DIR / "error_state.png", full_page=True)
            print("Saved error state to error_state.png")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
