import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# summary_info用于记录各特征可取值，格式即为'特征名':可取值个数
summary_info={
            "having_IP_Address" : { -1,1 },
            "URL_Length" : { 1,0,-1 },
            "Shortining_Service" : { 1,-1 },
            "having_At_Symbol" : { 1,-1 },
            "double_slash_redirecting" : { -1,1 },
            "Prefix_Suffix" : { -1,1 },
            "having_Sub_Domain" : { -1,0,1 },
            "SSLfinal_State" : { -1,1,0 },
            "Domain_registeration_length" : { -1,1 },
            "Favicon" : { 1,-1 },
            "port" : { 1,-1 },
            "HTTPS_token" : { -1,1 },
            "Request_URL" : { 1,-1 },
            "URL_of_Anchor" : { -1,0,1 },
            "Links_in_tags" : { 1,-1,0 },
            "SFH" : { -1,1,0 },
            "Submitting_to_email" : { -1,1 },
            "Abnormal_URL" : { -1,1 },
            "Redirect" : { 0,1 },
            "on_mouseover" : { 1,-1 },
            "RightClick" : { 1,-1 },
            "popUpWidnow" : { 1,-1 },
            "Iframe" : { 1,-1 },
            "age_of_domain" : { -1,1 },
            "DNSRecord" : { -1,1 },
            "web_traffic" : { -1,0,1 },
            "Page_Rank" : { -1,1 },
            "Google_Index" : { 1,-1 },
            "Links_pointing_to_page" : { 1,0,-1 },
            "Statistical_report" : { -1,1 },
            "Result" : { -1,1 }
        }

phishing_original = pd.read_csv("../Datas/PhishingWebsites.arff", 
                        names=summary_info.keys(),dtype=np.int8,engine='c')
phishing_data = np.array(phishing_original)

pca = PCA(n_components=20)
pca.fit(phishing_data)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)
