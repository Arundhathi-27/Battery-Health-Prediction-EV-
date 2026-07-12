def get_recommendation(status, soh):
    if status == "Healthy":
        return "Battery in good condition. Continue normal usage and monitor periodically."
    elif status == "Degrading":
        return "Battery showing wear. Avoid fast charging, reduce deep discharges, monitor temperature closely."
    elif status == "Critical":
        return "Battery health critical. Plan for replacement soon. Avoid full discharge and high current draw."
    else:
        return "Status unknown. Please check input readings."