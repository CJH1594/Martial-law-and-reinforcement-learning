import pandas as pd
import os

def preprocess_data_with_common_dates():
    # 파일 경로 설정
    data_dir = os.path.join(os.getcwd(), "data")
    stock_file = os.path.join(data_dir, "stock.csv")
    bond_file = os.path.join(data_dir, "bond.csv")
    exchange_file = os.path.join(data_dir, "exchange.csv")
    output_file = os.path.join(data_dir, "processed_data.csv")

    # 데이터 로드
    try:
        stock_data = pd.read_csv(stock_file)
        bond_data = pd.read_csv(bond_file, header=None, names=['date', 'bond_yield'])
        exchange_data = pd.read_csv(exchange_file)
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return
    
    # 쉼표 제거
    for data in [stock_data, exchange_data]:
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = data[column].str.replace(',', '', regex=False)

    # 컬럼 이름 정리
    stock_data.rename(columns={
        '날짜': 'date',
        '외국인 순매수': 'foreign_volume',
        '연기금 순매수': 'fund_volume',
        '개인 순매수': 'stock_individual_net_buy',
        '체결가': 'stock_price'
    }, inplace=True)

    exchange_data.rename(columns={
        '날짜': 'date',
        '매매기준율': 'exchange_rate'
    }, inplace=True)

    bond_data['bond_yield'] = bond_data['bond_yield'].astype(float)

    # 날짜 변환
    try:
        stock_data['date'] = pd.to_datetime(stock_data['date'], format='%y.%m.%d', errors='coerce')

        # Bond 데이터의 날짜 형식 처리
        bond_data['date'] = bond_data['date'].apply(lambda x: f"2024.{x}" if isinstance(x, str) else x)
        bond_data['date'] = pd.to_datetime(bond_data['date'], format='%Y.%m.%d.', errors='coerce')

        exchange_data['date'] = pd.to_datetime(exchange_data['date'], format='%Y.%m.%d', errors='coerce')

    except Exception as e:
        print(f"날짜 변환 중 오류 발생: {e}")
        return

    # NaT 제거
    stock_data = stock_data.dropna(subset=['date'])
    bond_data = bond_data.dropna(subset=['date'])
    exchange_data = exchange_data.dropna(subset=['date'])

    # 공통 날짜만 필터링
    common_dates = set(stock_data['date']).intersection(set(bond_data['date'])).intersection(set(exchange_data['date']))
    if not common_dates:
        print("공통 날짜가 없습니다. 데이터 범위를 확인하세요.")
        print(f"Stock 데이터 날짜 범위: {stock_data['date'].min()} ~ {stock_data['date'].max()}")
        print(f"Bond 데이터 날짜 범위: {bond_data['date'].min()} ~ {bond_data['date'].max()}")
        print(f"Exchange 데이터 날짜 범위: {exchange_data['date'].min()} ~ {exchange_data['date'].max()}")
        return

    stock_data = stock_data[stock_data['date'].isin(common_dates)]
    bond_data = bond_data[bond_data['date'].isin(common_dates)]
    exchange_data = exchange_data[exchange_data['date'].isin(common_dates)]

    # 데이터 병합
    try:
        merged_data = stock_data.merge(bond_data, on='date', how='inner')
        merged_data = merged_data.merge(exchange_data, on='date', how='inner')
    except Exception as e:
        print(f"데이터 병합 중 오류 발생: {e}")
        return
    
    # stock_individual_net_buy 컬럼 제거
    if 'stock_individual_net_buy' in merged_data.columns:
        merged_data = merged_data.drop(columns=['stock_individual_net_buy'])

    # 결과 저장
    try:
        merged_data.to_csv(output_file, index=False)
        print(f"공통 날짜 데이터를 포함한 파일이 저장되었습니다: {output_file}")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    preprocess_data_with_common_dates()
