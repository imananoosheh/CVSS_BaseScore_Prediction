# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:24:46 2020

@author: iman anooshehpour
"""


import json
import os

#   Version at the end of each file name
VERSION_NUM = "5.0"

#   getting .JSON file from current working directory (cwd)
file_location_2020 = os.path.join(os.getcwd(), "data", "nvdcve-1.1-2020.json")
jsonfile = open(file_location_2020, encoding="utf8")
cve_data = json.loads(jsonfile.read())
jsonfile.close()

#   Writting heading of the .JSON file
with open('train_data_cpe_V' + VERSION_NUM + '.csv', 'a') as train_data_file_heading:
    #  |                                           Input                                              |   Output  |
    #  | CVE_ID | CWE_ID | No of CPE_App | No of CPE_OS | No of CPE_Hardware | No of Vendors Invovled | baseScore |
    train_data_file_heading.write('AV,AC,PR,UI,Scope,CI,II,AI,baseScore' + '\n')


def write_to_csv(baseScoreMetrics_string, baseScore):
    with open('train_data_cpe_V' + VERSION_NUM + '.csv', 'a') as train_data_file:
        train_data_file.write(baseScoreMetrics_string + ',' + str(baseScore) + '\n')

counter=0

for cve_items in cve_data['CVE_Items']:
    if len(cve_items['impact']) != 0:
        string = cve_items['impact']['baseMetricV3']['cvssV3']['vectorString']
        #Exploitability Metrics
        attackVector_value = string.split('/')[1].split(':')[1]
        attackComplexity_value = string.split('/')[2].split(':')[1]
        privilegesRequired_value = string.split('/')[3].split(':')[1]
        userInteraction_value = string.split('/')[4].split(':')[1]
        scope_value = string.split('/')[5].split(':')[1]
        exploitabilityScore_string = str(attackVector_value) + ',' + str(attackComplexity_value) + ',' + str(privilegesRequired_value) + ',' + str(userInteraction_value) + ',' + str(scope_value) + ','
        
        #Impact Metrics
        confidentialityImpact_value = string.split('/')[6].split(':')[1]
        integrityImpact_value = string.split('/')[7].split(':')[1]
        availabilityImpact_value = string.split('/')[8].split(':')[1]
        impactScore_string = str(confidentialityImpact_value) + ',' + str(integrityImpact_value) + ','  + str(availabilityImpact_value)
        
        baseScoreMetrics_string = exploitabilityScore_string + impactScore_string
        #   Getting 3 number of baseScore, exploitabilityScore and impactScore for each CVE
        baseScore = cve_items['impact']['baseMetricV3']['cvssV3']['baseScore']   
        #   Here all parameter will be wrote for each cve in a line by calling the following function
        write_to_csv(baseScoreMetrics_string, baseScore)
    
    else:
        counter+=1


print("No of CVE items recorder in 2020 is "+ str(len(cve_data['CVE_Items']) - counter) + " out of " + str(len(cve_data['CVE_Items'])))
print("Not recorded: " + str(counter))
print("2020 version of json has been processed.")


#   getting .JSON file from current working directory (cwd)
file_location_2019 = os.path.join(os.getcwd(), "data", "nvdcve-1.0-2019.json")
jsonfile = open(file_location_2019, encoding="utf8")
cve_data = json.loads(jsonfile.read())
jsonfile.close()


counter = 0

for cve_items in cve_data['CVE_Items']:
    if len(cve_items['impact']) != 0:
        string = cve_items['impact']['baseMetricV3']['cvssV3']['vectorString']
        #Exploitability Metrics
        attackVector_value = string.split('/')[1].split(':')[1]
        attackComplexity_value = string.split('/')[2].split(':')[1]
        privilegesRequired_value = string.split('/')[3].split(':')[1]
        userInteraction_value = string.split('/')[4].split(':')[1]
        scope_value = string.split('/')[5].split(':')[1]
        exploitabilityScore_string = str(attackVector_value) + ',' + str(attackComplexity_value) + ',' + str(privilegesRequired_value) + ',' + str(userInteraction_value) + ',' + str(scope_value) + ','
        
        #Impact Metrics
        confidentialityImpact_value = string.split('/')[6].split(':')[1]
        integrityImpact_value = string.split('/')[7].split(':')[1]
        availabilityImpact_value = string.split('/')[8].split(':')[1]
        impactScore_string = str(confidentialityImpact_value) + ',' + str(integrityImpact_value) + ','  + str(availabilityImpact_value)
        baseScoreMetrics_string = exploitabilityScore_string + impactScore_string
        #   Getting 3 number of baseScore, exploitabilityScore and impactScore for each CVE
        baseScore = cve_items['impact']['baseMetricV3']['cvssV3']['baseScore']
        #   Here all parameter will be wrote for each cve in a line by calling the following function
        write_to_csv(baseScoreMetrics_string, baseScore)
              
    else:
        counter+=1
    

print("No of CVE items recorder in 2019 is "+ str(len(cve_data['CVE_Items']) - counter) + " out of " + str(len(cve_data['CVE_Items'])))
print("Not recorded: " + str(counter))
print("2019 version of json has been processed.")

#   getting .JSON file from current working directory (cwd)
file_location_2018 = os.path.join(os.getcwd(), "data", "nvdcve-1.0-2018.json")
jsonfile = open(file_location_2018, encoding="utf8")
cve_data = json.loads(jsonfile.read())
jsonfile.close()


counter = 0

for cve_items in cve_data['CVE_Items']:
    if len(cve_items['impact']) != 0:
        string = cve_items['impact']['baseMetricV3']['cvssV3']['vectorString']
        #Exploitability Metrics
        attackVector_value = string.split('/')[1].split(':')[1]
        attackComplexity_value = string.split('/')[2].split(':')[1]
        privilegesRequired_value = string.split('/')[3].split(':')[1]
        userInteraction_value = string.split('/')[4].split(':')[1]
        scope_value = string.split('/')[5].split(':')[1]
        exploitabilityScore_string = str(attackVector_value) + ',' + str(attackComplexity_value) + ',' + str(privilegesRequired_value) + ',' + str(userInteraction_value) + ',' + str(scope_value) + ','
        #Impact Metrics
        confidentialityImpact_value = string.split('/')[6].split(':')[1]
        integrityImpact_value = string.split('/')[7].split(':')[1]
        availabilityImpact_value = string.split('/')[8].split(':')[1]
        impactScore_string = str(confidentialityImpact_value) + ',' + str(integrityImpact_value) + ','  + str(availabilityImpact_value)
        baseScoreMetrics_string = exploitabilityScore_string + impactScore_string
        #   Getting 3 number of baseScore, exploitabilityScore and impactScore for each CVE
        baseScore = cve_items['impact']['baseMetricV3']['cvssV3']['baseScore']  
        #   Here all parameter will be wrote for each cve in a line by calling the following function
        write_to_csv(baseScoreMetrics_string, baseScore)
    
    else:
        counter+=1


print("No of CVE items recorder in 2018 is "+ str(len(cve_data['CVE_Items']) - counter) + " out of " + str(len(cve_data['CVE_Items'])))
print("Not recorded: " + str(counter))
print("2018 version of json has been processed.")