DATA_CONFIG = {
    "datasets": [
        {
            "name": "IoTID20",
            "path": "data/IoTID20_final_max_1M.csv",
            "dropped_cols": [
                "Flow_ID",
                "Timestamp",
                "Src_IP",
                "Dst_IP",
                "Cat",
                "Sub_Cat",
            ],
            "target": "Target",
            "classes": ["Benign", "Malicious"],
        },
        {
            "name": "CIC-DDoS-2019",
            "path": "data/CIC-DDoS-2019_final_max_1M.csv",
            "dropped_cols": [
                "Flow ID",
                "Source IP",
                "Destination IP",
                "Timestamp",
                "SimillarHTTP",
            ],
            "target": "Target",
            "classes": ["Benign", "Malicious"],
        },
        # drop this dataset due to extreme imbalance
        {
            "name": "BOT-IoT",
            "path": "data/BOT-IoT_final_max_1M.csv",
            "dropped_cols": [
                "flgs",
                "proto",
                "saddr",
                "daddr",
                "state",
                "category",
                "subcategory",
            ],
            "target": "Target",
            "classes": ["Benign", "Malicious"],
        },
        {
            "name": "CIC-IDS-2017",
            "path": "data/CIC-IDS-2017_final_max_1M.csv",
            "dropped_cols": ["Flow ID", "Source IP", "Destination IP", "Timestamp"],
            "target": "Target",
            "classes": ["Benign", "Malicious"],
        },
        {
            "name": "CIC-IDS-2018",
            "path": "data/CIC-IDS-2018_final_max_1M.csv",
            "dropped_cols": ["Timestamp"],
            "target": "Target",
            "classes": ["Benign", "Malicious"],
        },
    ],
}
