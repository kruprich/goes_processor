from datetime import datetime, timedelta

def generate_prefixes(start_date, end_date):
    print("Generating Prefixes")
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    prefixes = []
    while start_dt <= end_dt:
        year = start_dt.year
        julian_day = start_dt.strftime('%j')
        prefixes.append(f"GLM-L2-LCFA/{year}/{julian_day}/")
        start_dt += timedelta(days=1)
    print('prefixes complete')
    return prefixes