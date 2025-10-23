import pandas as pd
import numpy as np

from google.cloud import bigquery
from google.oauth2 import service_account

import os
import shutil
from tenacity import retry, stop_after_attempt, wait_fixed

import requests
import utils.ssl_setup as ssl
from ftfy import fix_encoding

from PIL import ImageFile, Image
import imgkit
import imagesize
import utils.img_processing as improc
import json
import re
from bs4 import BeautifulSoup

import pyperclip


def empty_folder(html_path, original_img_path, transformed_img_path, text_folder_path):
    confirm = input("You are about to delete img_staging sub-folders. Proceed? (y/n)")
    if confirm == 'y':
        for path in [html_path, original_img_path, transformed_img_path, text_folder_path]:
            shutil.rmtree(path)
            os.makedirs(path)
    
    return

yu7=gt
def clean_bo_report(bo_report_path):
    df = pd.read_csv(bo_report_path, skiprows=5)

    df['Test'] = np.where(df['Campaign'].str.lower().str.contains('|'.join(['test', 'dry'])), 1, 0)
    df = df[df['Test'] == 0]

    market = {
        'AUSTRALIA':'AU',
        'THAILAND':'TH',
        'SINGAPORE':'SG',
        'INDONESIA':'ID',
        'MALAYSIA':'MY',
        'VIETNAM':'VN',
        'PHILIPPINES':'PH',
        'NEW ZEALAND':'NZ'
    }

    df['market'] = df['Market Area'].replace(market)
    df['cid'] = '0000' + df['Campaign Id'].astype(str)

    df = df[['cid', 'market']]

    return df


def get_assetid(cid, cookie):

    url = "https://www.global-cdm.net/sap/opu/odata/sap/CUAN_CAMPAIGN_SRV/$batch"  

    querystring = {"sap-client":"100"}

    payload = f"--batch_1\nContent-Type: application/http\nContent-Transfer-Encoding: binary\n\nGET C_MKT_CampaignAssetsDetailAll?sap-client=100&$skip=0&$top=100&$orderby=CampaignAssetLastChgdDateTime%20desc&$filter=CampaignID%20eq%20%27{cid}%27&$inlinecount=allpages HTTP/1.1\nsap-cancel-on-close: true\nsap-contextid-accept: header\nAccept: application/json\nAccept-Language: en\nDataServiceVersion: 2.0\nMaxDataServiceVersion: 2.0\nx-csrf-token: -7VtTuIxpNfydmyd-eqEgA==\n\n\n--batch_1--\n"
    headers = {
        "cookie": "UqZBpD3n3iPIDwJU=v1W8ckg1guDYU",
        "Content-Type": "multipart/mixed;boundary=batch_1",
        "Cookie": f"{cookie}"
    }

    try:
        response = ssl.get_legacy_session().post(url, data=payload, headers=headers, params=querystring)
        response_text = response.text
    
        campaign_id_match = re.search(r"CampaignID=\\?'(\d+)\\?'", response_text)
        campaign_assets = re.findall(r"CampaignAsset=\\?'(\d+)\\?'", response_text)
        if campaign_id_match:
            # campaign_id = campaign_id_match.group(1)
            if campaign_assets:
                latest_asset = campaign_assets[0]  # Assuming the response is ordered by date
                return latest_asset
        else:
            print(f"No Asset ID found in response for campaign_id {cid}")
            return None
    except requests.RequestException as e:
        print(f"Request failed for campaign_id {cid}: {e}")
        return None


def get_df_assetid(df, asset_id_path, cookie):
    cids = df['cid'].to_list()
    assets = {}

    for cid in cids:
        assets[cid] = get_assetid(cid, cookie)

    df['asset_id'] = df['cid'].map(assets)

    df['img_name'] = df['cid'] + '_' + df['market'] + '.jpg'

    df.to_parquet(asset_id_path, index = False)

    return


def get_html(asset_id, cookie):
    url = "https://www.global-cdm.net/sap/opu/odata/sap/CUAN_MARKETING_CPG_MESSAGE_SRV/$batch"

    querystring = {"sap-client":"100"}

    payload = f"--batch_1\nContent-Type: application/http\nContent-Transfer-Encoding: binary\n\nGET MarketingEngagements({asset_id})?sap-client=100&$expand=ToBlockContent%2cToBlockContent%2fToConditionBlockContent%2cToCondition%2cToCondition%2fConditionItem%2cAuthorizationUser%2cToAgencies%2cToBlockContent%2fToReuseCondition%2cToBlockContent%2fToReuseCondition%2fToReuseConditionItem%2cToTag%2cFeatures%2cDataSourceAttribute HTTP/1.1\nsap-cancel-on-close: true\nsap-contextid-accept: header\nAccept: application/json\nAccept-Language: en\nDataServiceVersion: 2.0\nMaxDataServiceVersion: 2.0\n\n\n--batch_1--\n"
    headers = {
        "cookie": "UqZBpD3n3iPIDwJU=v1W8ckg1guDYU",
        "Content-Type": "multipart/mixed;boundary=batch_1",
        "Cookie": f"{cookie}"
    }

    response = ssl.get_legacy_session().post(url, data=payload, headers=headers, params=querystring)
    response_text = response.text

    firstValue = response_text.index("{")
    lastValue = len(response_text) - response_text[::-1].index("}")

    jsonString = response_text[firstValue:lastValue + 1]
    data = json.loads(jsonString)

    html = data['d']['ToBlockContent']['results'][2]['Content']
    html = fix_encoding(html) #encoding utf-8
    html = html.replace('\n', '').replace('\t', '')

    return html


def remove_ptag(html):
    soup = BeautifulSoup(html, 'html.parser')

    ptags = soup.find_all('p')
    for tag in ptags:
        tag.decompose()
    
    return str(soup) #return html object


def convert_html_to_img(html_file_path, img_file_path):
    #convert html to img: https://pypi.org/project/imgkit/
    imgkit.from_file(html_file_path, img_file_path, options={'encoding': "UTF-8", 'quiet': ''})

    return


def process_img(img_file_path, transformed_img_path, img_name):
    img = Image.open(img_file_path)

    width, height = img.size
    crop_pixel = (10, 0, width-10, height)

    cropped_img = improc.crop(img, crop_pixel)
    recropped_img = improc.recrop(cropped_img, img)
    resized_img = improc.resize(recropped_img)
    improc.save_img(resized_img, transformed_img_path, img_name=img_name)

    # resized_width, resized_height = resized_img.size

    return #resized_height / resized_width #image ratio


@retry(stop=stop_after_attempt(5))
def get_img_size(url):
    resume_header = {'Range': 'bytes=0-2000000'}    ## the amount of bytes you will download
    data = requests.get(url, stream=True, headers=resume_header).content

    p = ImageFile.Parser()
    p.feed(data)    ## feed the data to image parser to get photo info from data headers
    if p.image:
        return p.image.size ## return width, height


def get_pod_height(html, cid):
    soup = BeautifulSoup(html, 'html.parser')

    # Remove all mobile tags
    mobile_tags_1 = soup.find_all('tr', class_='es-desk-hidden')
    mobile_tags_2 = soup.find_all('table', class_='es-desk-hidden')
    for tag in mobile_tags_1:
        tag.decompose()
    for tag in mobile_tags_2:
        tag.decompose()

    # Filter all tags with image
    td_img = []
    for td in soup.find_all('td'):
        if td.find_all('td'):
            continue #skip parent td tags
        if td.find_all('img'):
            td_img.append(td)

    # Filter valid img tags
    imgs = []
    for img in td_img:
        img_object = img.find_all('img')[0]
        pod_width = img_object.get('width', -1)
        if pod_width == '100%':
            pod_width = '2'
        img_dict = {
            'url' : img_object.get('src', ''),
            'width' : int(pod_width if str(pod_width).isdigit() else -1),
        }

        # print(int(img_object.get('width', -1) if img_object.get('width', -1).isdigit() else -1))
        # print(img_object.get('width', -1))

        links = img.find_all('a')
        if links:
            img_dict['name'] = links[0].get('name', '')
        else:
            img_dict['name'] = ''
        
        if '.' not in img_dict['url'].split('/')[-1]:
            continue #skip img without extension
        elif img_dict['width'] <= 1: #img must have width
            continue
        # elif img_dict['name'].startswith('999'):
        #     continue #skip footers
        else:
            imgs.append(img_dict)

    # Pod height dict
    pods = {'HYBRIS_ID':[], 'Pod':[], 'Pod_adj':[], 'Height':[], 'Url':[]}

    scenarios = {
        '1A' : "Normal. Update prev_pod. Add to pods dict.",
        '1B' : "No label. Combine height with next image.",
        '1C' : "Same pod No as previous img. Combine height with previous image.",
        '2A' : "Side-by-side. Define pod_adj (equal prev_pod) and add to pods dict (at the start to avoid 1C, height = 0).",
        '2B' : "Side-by-side no label. Do nothing.",
        '2C' : "Side-by-side same pod No as previous img. Do nothing."
    } #scenarios reference

    prev_pod = ''
    max_width = max([img['width'] for img in imgs])
    # width_actual = 0
    side_by_side_width = 0
    height_to_be_combined = 0

    for i, img in enumerate(imgs):
        pod_no = img['name'][0:2].strip('_')

        # Check scenarios
        if img['name'] == '':
            code_1 = 'B'
        elif pod_no == prev_pod:
            code_1 = 'C'
        elif pod_no == '99':
            code_1 = 'C'
        else:
            code_1 = 'A'

        if img['width'] == 2: #special code for width = 100%
            code_2 = '1'
            side_by_side_width = 0 #reset
        elif img['width'] < max_width:
            try:
                if side_by_side_width == 0:
                    if img['width'] + imgs[i+1]['width'] <= max_width: #check if image is actually side-by-side, or just slightly smaller than max_width
                        side_by_side_width = img['width'] #start of side-by-side
                        code_2 = '1'
                    else: #false alarm, not side-by-side
                        code_2 = '1'
                else: #already detecting side-by-side sequence
                    if img['width'] + side_by_side_width <= max_width:
                        side_by_side_width += img['width'] #continue side-by-side
                        code_2 = '2'
                    elif img['width'] + imgs[i+1]['width'] <= max_width: #check if image is actually side-by-side, or just slightly smaller than max_width
                        side_by_side_width = img['width'] #start of another side-by-side row, reset width count
                        code_2 = '1'
                    else: #false alarm, not side-by-side
                        code_2 = '1'
                        side_by_side_width = 0
            except: #catch last pod
                code_2 = '1'
                side_by_side_width = 0
        else:
            code_2 = '1'
            side_by_side_width = 0 #reset

        # Action for each scenario
        if code_2 == '1':
            try:
                width, height = get_img_size(img['url'])
                # if width_actual == 0:
                #     width_actual = width
            except:
                height = 0
                print(f"{img['url']} download failed!")
            
            if code_1 == 'A':
                pods['Pod'].append(pod_no)
                pods['Pod_adj'].append(pod_no)
                pods['Height'].append(height + height_to_be_combined)
                pods['Url'].append(img['url'])
                prev_pod = pod_no
                height_to_be_combined = 0 #reset
            elif code_1 == 'B':
                height_to_be_combined += height
            else:
                pods['Height'][-1] += height
        else:
            if code_1 == 'A':
                pods['Pod'].insert(0, pod_no)
                pods['Pod_adj'].insert(0, prev_pod)
                pods['Height'].insert(0, 0)
                pods['Url'].insert(0, '')
    
    pods['Height'][-1] += height_to_be_combined #if any left to combine
    
    pods['Pod'].insert(0, '99')
    pods['Pod_adj'].insert(0, '99')
    pods['Height'].insert(0, 0)
    pods['Url'].insert(0, '')
    pods['HYBRIS_ID'] = [cid] * len(pods['Pod'])

    return pods#, width_actual


def store_df_image(asset_id_path, html_path, original_img_path, transformed_img_path, pod_height_path, cookie):    
    df = pd.read_parquet(asset_id_path)

    cids, asset_ids, markets = df['cid'].to_list(), df['asset_id'].to_list(), df['market'].to_list()

    pod_heights = []
    for cid, asset_id, market in zip(cids, asset_ids, markets):
        try:
            html = get_html(asset_id, cookie)
            html = remove_ptag(html)

            html_file = os.path.join(html_path, f'{cid}_{market}.html')
            with open(html_file, 'w', encoding='utf-8') as file:
                file.write(html)

        except:
            print(f'Fail to fetch HTML for {cid}_{market}')
            continue

        try:
            img_file = os.path.join(original_img_path, f'{cid}_{market}.jpg')
            convert_html_to_img(html_file, img_file)
            process_img(img_file, transformed_img_path, f'{cid}_{market}.jpg')
        except:
            print(f'Fail to convert HTML to image for {cid}_{market}')
            continue

        try:
            pod_height = get_pod_height(html, cid)

            to_df = pd.DataFrame(pod_height)
            to_df = to_df.drop_duplicates(subset=['HYBRIS_ID', 'Pod'])
            # to_df['Img_Height'] = max_width_actual * img_ratio #image height actual
            to_df['Total_Pods'] = to_df['Pod_adj'].nunique()
            to_df['Total_Pod_Height'] = to_df['Height'].sum()
            to_df['Img_Height'] = to_df['Height'].sum()*1.1 #add 10% for footer
            # to_df['pod_top'] = to_df['Pod_adj'].min()
            # to_df['pod_bot'] = to_df[to_df['Pod_adj'] != '999']['Pod_adj'].max()
            pod_heights.append(to_df)
        except:
            print(f'Fail to get pod height for {cid}_{market}')
            continue

    pod_height_df = pd.concat(pod_heights)
    pod_height_df.to_parquet('test.parquet', index = False)

    pod_height_df['Pod_Start_Pixel'] = pod_height_df.groupby('HYBRIS_ID')['Height'].transform(lambda x: x.shift().fillna(0).cumsum())
    pod_height_df['Pod_Position'] = np.where(pod_height_df['Pod_adj'] == '99', 'Footers',
                                             np.where(pod_height_df['Pod_Start_Pixel'] <= pod_height_df['Total_Pod_Height'] * 0.25, 'Top',
                                                      np.where(pod_height_df['Pod_Start_Pixel'] > pod_height_df['Total_Pod_Height'] * 0.75, 'Bottom', 'Middle')))
    df_temp = pod_height_df[pod_height_df['Pod'] == pod_height_df['Pod_adj']][['HYBRIS_ID', 'Pod_adj', 'Pod_Position']]
    pod_height_df = pod_height_df.merge(df_temp, on=['HYBRIS_ID', 'Pod_adj'], how='left').rename(columns={'Pod_Position_x':'Pod_Position', 'Pod_Position_y':'Mapper'})
    pod_height_df['Pod_Position'] = np.where(pod_height_df['Pod'] == pod_height_df['Pod_adj'], pod_height_df['Pod_Position'], pod_height_df['Mapper'])

    pod_height_df['Footer_Height'] = pod_height_df['Img_Height'] - pod_height_df['Total_Pod_Height']
    pod_height_df['Height'] = np.where(pod_height_df['Pod'] == '99', pod_height_df['Footer_Height'], pod_height_df['Height'])
    pod_height_df['Height_pct'] = pod_height_df['Height'] / pod_height_df['Img_Height']
    pod_height_df['height_pct_bin'] = pd.cut(pod_height_df['height_pct'], bins=[0,0.05,0.1,0.15,0.2,0.5,1], labels=['0-5%', '05-10%', '10-15%', '15-20%', '20-50%', '50-100%'])

    for col in ['Pod', 'Pod_adj', 'Height', 'Total_Pods', 'Total_Pod_Height', 'Img_Height']:
        try:
            pod_height_df[col] = pd.to_numeric(pod_height_df[col], errors='coerce').astype('Int64')
        except TypeError as e:
            pod_height_df[col] = np.floor(pod_height_df[col]).astype('Int64')
    pod_height_df[['HYBRIS_ID', 'Pod', 'Pod_adj', 'Height', 'Height_pct', 'Pod_Position', 'Total_Pods', 'Total_Pod_Height', 'Img_Height', 'Url']].to_parquet(pod_height_path, index = False)

    return


def process_click_report(click_report_path, pod_height_path, unmatched_path):
    df = pd.read_csv(click_report_path)
    df = df[['Campaign Id', 'Country Id', 'Schedule Date', 'Contents Label', 'Delivered', 'Opened', 'Label Clicks']]
    df.columns = ['HYBRIS_ID', 'Market_Area', 'date', 'Contents_Label', 'Delivered', 'Opened', 'Label_Clicks']

    df['HYBRIS_ID'] = '0000' + df['HYBRIS_ID'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

    df['Pod'] = pd.to_numeric(df['Contents_Label'].str.split('_').str[0], errors='coerce').astype('Int64')
    df['Pod'] = df['Pod'].fillna(-1) #because np.where requires non-null column
    df['Pod'] = np.where(df['Pod'] == 999, 99, df['Pod'])
    df['Pod'] = df['Pod'].replace(-1, pd.NA)
    df = df[df['Pod'].notna()]

    # Pods for mobile version starts at 50
    df['Pod'] = np.where((df['Pod'] >= 50) & (df['Pod'] < 99), df['Pod'] - 50, df['Pod'])

    #Extract specificed text from content label 
    df['Label_Name'] = df['Contents_Label'].str.split('_').str[2]

    #Remove patterns from Label_Name Column
    patterns_to_remove = ("mobile", "desktop") #Add patterns in this tuple
    df['Label_Name'] = np.where(df['Label_Name'].str.startswith(patterns_to_remove), df['Label_Name'].str.split('-').str[1:].apply('-'.join), df['Label_Name'])

    # Find total click (non-unique) for each campaign
    df['Total_Clicks'] = df.groupby('HYBRIS_ID')['Label_Clicks'].transform('sum')

    df['Click_Rate'] = df['Label_Clicks'] / df['Total_Clicks']
    df['CTR'] = df['Label_Clicks'] / df['Opened']
    df['Sent_CTR'] = df['Label_Clicks'] / df['Delivered']
    for col in ['Click_Rate', 'CTR', 'Sent_CTR']:
        df[col] = df[col].fillna(0)

    # Make Contents Label unique just in case
    df = df.groupby(['HYBRIS_ID', 'date', 'Market_Area', 'Contents_Label', 'Pod', 'Label_Name'], as_index=False)[['Label_Clicks', 'Click_Rate', 'CTR', 'Sent_CTR']].sum()

    # Add Unsubscribe column
    df['Unsubscribe_Flag'] = df['Contents_Label'].apply(lambda x: 1 if 'unsub' in x.lower() else 0)

    # Add Filtered_Clicks and Filtered_Clicks_Unsub columns
    df['Filtered_Clicks'] = df.apply(lambda x: x['Label_Clicks'] if '999' not in x['Contents_Label'] else 0, axis=1)
    df['Filtered_Clicks_Unsub'] = df.apply(lambda x: x['Label_Clicks'] if '999' not in x['Contents_Label'] or x['Unsubscribe_Flag'] == 1 else 0, axis=1)

    df['Total_Filtered_Clicks'] = df.groupby('HYBRIS_ID')['Filtered_Clicks'].transform('sum')
    df['Total_Filtered_Clicks_Unsub'] = df.groupby('HYBRIS_ID')['Filtered_Clicks_Unsub'].transform('sum')

    # Calculate CR_Excl_Footer and CR_With_Unsubscribe
    df['CR_Excl_Footer'] = df.apply(
        lambda x: 0 if '999' in x['Contents_Label'] else (
            0 if x['Total_Filtered_Clicks'] == 0 else x['Label_Clicks'] / x['Total_Filtered_Clicks']
        ), axis=1
    )

    df['CR_With_Unsubscribe'] = df.apply(
        lambda x: 0 if '999' in x['Contents_Label'] and x['Unsubscribe_Flag'] == 0 else (
            0 if x['Total_Filtered_Clicks_Unsub'] == 0 else x['Label_Clicks'] / x['Total_Filtered_Clicks_Unsub']
        ), axis=1
    )

    df['CTR_With_Unsubscribe'] = df.apply(
        lambda x: x['CTR'] if x['Pod'] != 99 or x['Unsubscribe_Flag'] == 1 else 0, axis=1
    )

    df['Sent_CTR_With_Unsubscribe'] = df.apply(
        lambda x: x['Sent_CTR'] if x['Pod'] != 99 or x['Unsubscribe_Flag'] == 1 else 0, axis=1
    )

    pod_height = pd.read_parquet(pod_height_path)
    errors_cid = pod_height[(pod_height['Height'] < 0) | (pod_height['Pod'].between(50,55))]['HYBRIS_ID'].unique()
    pod_height = pod_height[~pod_height['HYBRIS_ID'].isin(errors_cid)]
    # pod_height['Pod'] = np.where(pod_height['Pod'] == 999, 99, pod_height['Pod'])
    # pod_height['Pod_adj'] = np.where(pod_height['Pod_adj'] == 999, 99, pod_height['Pod_adj'])
    
    #test
    df.to_parquet('testdf.parquet', index = False)
    pod_height.to_parquet('testpod.parquet', index = False)
    
    df = df.merge(pod_height, on=['HYBRIS_ID', 'Pod'], how='left')
    df_unmatched = df[df['Pod_adj'].isna()]
    df_unmatched.to_parquet(unmatched_path, index=False)

    df = df[df['Pod_adj'].notna()]
    df['height_pct_bin'] = pd.cut(df['height_pct'], bins=[0,0.05,0.1,0.15,0.2,0.5,1], labels=['0-5%', '05-10%', '10-15%', '15-20%', '20-50%', '50-100%'])
    df = df[['HYBRIS_ID', 'date', 'Market_Area', 'Contents_Label', 'Label_Clicks', 'Click_Rate', 'CTR', 'Sent_CTR', 'Label_Name', 'Pod', 'Pod_adj', 'Pod_Position', 'Total_Pods', 'Height', 'Height_pct', 'Height_pct_bin', 'Url', 'Unsubscribe_Flag', 'CR_Excl_Footer', 'CR_With_Unsubscribe', 'CTR_With_Unsubscribe', 'Sent_CTR_With_Unsubscribe']]

    return df


def update_bq_click_report_table(df_click_report, service_account_json):
    from google.cloud import bigquery
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    client = bigquery.Client(project = 'xyz', credentials=credentials)

    tbl_click_report = client.dataset('gcdm').table('click_report_temp')

    load_job = client.load_table_from_dataframe(df_click_report, tbl_click_report)
    load_job.result()

    # Merge with existing click_report table
    QUERY = (
        f"""

        MERGE `xxx.gcdm.click_report` T
        USING `xxx.gcdm.click_report_temp` S
        ON T.HYBRIS_ID = S.HYBRIS_ID AND T.Contents_Label = S.Contents_Label
        WHEN MATCHED THEN
        UPDATE SET date = S.date, Label_Clicks = S.Label_Clicks, Click_Rate = S.Click_Rate, CTR = S.CTR, Sent_CTR = S.Sent_CTR, Label_Name = S.Label_Name, Pod = S.Pod, Pod_adj = S.Pod_adj, Pod_Position = S.Pod_Position, Total_Pods = S.Total_Pods, Height = S.Height, Height_pct = S.Height_pct, Height_pct_bin = S.Height_pct_bin, Url = S.Url, Unsubscribe_Flag = S.Unsubscribe_Flag, CR_Excl_Footer = S.CR_Excl_Footer, CR_With_Unsubscribe = S.CR_With_Unsubscribe, CTR_With_Unsubscribe = S.CTR_With_Unsubscribe, Sent_CTR_With_Unsubscribe = S.Sent_CTR_With_Unsubscribe
        WHEN NOT MATCHED THEN
        INSERT ROW
        ;

        DROP TABLE IF EXISTS `xxx.gcdm.click_report_temp`
        ;

        """
    )

    query_job = client.query(QUERY)
    results = query_job.result()

    return


def extract_text(img_path, text_folder_path, link_to_tesseract):
    import utils.img_text_extract as imtext
    import utils.img_processing as improc

    img_dict = improc.list_img(img_path, open_img=True) #this will give a dict of {key = img file name : value = the opened image}

    for file_name, img in img_dict.items():
        market = file_name.split('.')[0].split('_')[-1]

        langs = {
        'VN': "vie+eng",
        'TH': "tha+eng"
        }
        default_lang = "eng"

        language = langs.get(market, default_lang)

        text = imtext.text_extract(img, lang=language, link_to_tesseract=link_to_tesseract)

        imtext.create_text_file(text, text_folder_path, image_name=file_name.split('.')[0])

    return


def qc_check(html_path, original_img_folder, transformed_img_folder, text_folder_path):

    def count_files(folder_path, extension):
        file_list = [f for f in os.listdir(folder_path) if f.endswith(extension)]
        return len(file_list)

    count_html = count_files(folder_path = html_path, extension = '.html')
    count_original = count_files(folder_path = original_img_folder, extension = '.jpg')
    count_transformed = count_files(folder_path = transformed_img_folder, extension = '.jpg')
    count_text = count_files(folder_path = text_folder_path, extension = '.txt')

    # Print counts
    print(f'original image files: {count_html}')
    print(f'original image files: {count_original}')
    print(f'Transformed image files: {count_transformed}')
    print(f'Text files: {count_text}')

    # Check for equality
    if count_original == count_transformed == count_text:
        print("Perfect, no error.")
    else:
        print("Error, please double-check.")

    return


def get_img_height(transformed_img_folder):
    import utils.img_processing as improc

    height_dict = {}

    img_dict = improc.list_img(transformed_img_folder, open_img=True) #this will give a dict of {key = img file name : value = the opened image}
    for file_name, img in img_dict.items():
        _, height = img.size
        height_dict[file_name] = height

    return height_dict


def update_bq_campaign_asset_table(asset_id_path, service_account_json):
    from google.cloud import bigquery
    from google.oauth2 import service_account

    df = pd.read_parquet(asset_id_path)
    df['HYBRIS_ID'] = df['cid'].astype(str)
            
    # df['height_px'] = df['img_name'].map(height_dict)
    # df_data_backup = df[['HYBRIS_ID', 'asset_id', 'height_px']]
    # df_data_backup.to_parquet(f'{data_backup_path}.parquet', index=False)

    # Upload campaign_asset
    df_upload_bq = df[['HYBRIS_ID', 'asset_id']]    

    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    client = bigquery.Client(project = 'xxx', credentials=credentials)
    
    tbl_campaign_asset = client.dataset('gcdm').table('campaign_asset_temp')

    load_job = client.load_table_from_dataframe(df_upload_bq, tbl_campaign_asset)
    load_job.result()

    # Merge with existing campaign_asset table
    QUERY = (
        f"""

        MERGE `xxx.gcdm.campaign_asset_edm` T
        USING `xxx.gcdm.campaign_asset_temp` S
        ON T.HYBRIS_ID = S.HYBRIS_ID
        WHEN NOT MATCHED THEN
        INSERT ROW
        ;

        DROP TABLE IF EXISTS `xxx.gcdm.campaign_asset_temp`
        ;

        """
    )

    query_job = client.query(QUERY)
    results = query_job.result()

    return


@retry(stop=stop_after_attempt(3), wait=wait_fixed(15))
def upload_to_bucket(bucket_name, blob_name, path_to_file, metadata=False, storage_client=None, service_account_json=None):
    from google.cloud import storage

    if service_account_json:
        storage_client = storage.Client.from_service_account_json(service_account_json)

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(path_to_file)

    if metadata:
        blob.metadata = metadata
        blob.patch()

    return


def update_bq_campaigns_table(bo_report_path, service_account_json):
    df = pd.read_csv(bo_report_path, skiprows=5)

    market = {
        'AUSTRALIA':'AU',
        'THAILAND':'TH',
        'SINGAPORE':'SG',
        'INDONESIA':'ID',
        'MALAYSIA':'MY',
        'VIETNAM':'VN',
        'PHILIPPINES':'PH',
        'NEW ZEALAND':'NZ'
    }
    df['Market Area'] = df['Market Area'].replace(market)

    #format str cols
    for col in ['Campaign Start', 'Campaign End', 'Executed Date', 'Campaign Id', 'Program ID', 'Target Group ID']:
        df[col] = df[col].astype(str).str.removesuffix('.0')

    #format int cols
    for col in ['Targeted Sent', 'Delivery Success', 'Opened/Displayed', 'Clicked', 'Unsubscription', 'Delivery Failed']:
        df[col] = df[col].str.replace(',', '').fillna(0).astype(int)

    df['HYBRIS_ID'] = '0000' + df['Campaign Id']
    df['date'] = pd.to_datetime(df['Executed Date'], format='%Y%m%d').dt.date

    df['Email Title'] = df['Email Title'].str.replace('\|.*\|', 'First Name', regex=True)
    df['Email Title'] = df['Email Title'].str.replace(r'\[QC\]\s*', '', regex=True) \
                                        .str.replace(r'QC\]\s*', '', regex=True)

    #identify and remove test campaigns
    df['Test'] = np.where(df['Campaign'].str.lower().str.contains('|'.join(['test', 'dry'])), 1, 0)
    df = df[df['Test'] == 0]

    df = df.drop(columns=['Campaign Name(2)', 'SMS Message Part', 'Push execute', 'Test'])
    df = df.rename(columns={'Campaign Description': 'Description'})
    df.rename(columns=lambda col: col.replace(' ', '_').replace('/', '_'), inplace=True)

    #Create dummy columns. These cols are populated later in BQ when doing campaign tagging
    df['Model'] = 'temporary'
    df['source'] = 'temporary'
    df['Clicked_Non_Unique'] = 0

    # Safety protocol to prevent duplication in source data
    df = df.drop_duplicates(subset=['HYBRIS_ID'], keep='last')
    df.columns

    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    client = bigquery.Client(project = 'xxx', credentials=credentials)

    tbl_campaigns = client.dataset('gcdm').table('campaigns_temp')

    load_job = client.load_table_from_dataframe(df, tbl_campaigns)
    load_job.result()

    # Merge with existing click_report table
    QUERY = (
        f"""

        MERGE `xxx.gcdm.campaigns` T
        USING `xxx.gcdm.campaigns_temp` S
        ON T.HYBRIS_ID = S.HYBRIS_ID
        WHEN NOT MATCHED THEN
        INSERT ROW
        ;

        DROP TABLE IF EXISTS `xxx.gcdm.campaigns_temp`
        ;

        """
    )

    query_job = client.query(QUERY)
    results = query_job.result()

    return