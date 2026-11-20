import streamlit as st

HOSPITALS_DATA = [
    {
        "name": "Khushi Pet Care",
        "phone": "+91 95359 61590",
        "address": "IV Stage 3 Block, Rajarajeshwari Nagar, Bengaluru, Karnataka 560098",
        "rating": 4.5,
        "reviews": 230,
        "open_close": "9:30 AM – 9:00 PM (Daily)",
        "website": "https://khushi-petcare.com",
        "map_url": "https://maps.app.goo.gl/aoE41d9LbrsDoJvm9",
        "image": "https://lh3.googleusercontent.com/p/AF1QipO4Uc2JuVDc-55Pox-aL4zw1jyEwly_vHFD0dQg=w408-h544-k-no"
    },
    {
        "name": "BRIGADE PET CLINIC AND GROOMING CENTER",
        "phone": "+91 78999 45896",
        "address": "BHCS Layout, Subramanyapura, Bengaluru, Karnataka 560061",
        "rating": 3.8,
        "reviews": 24,
        "open_close": "10:00 AM – 9:00 PM (Closed Tuesday)",
        "website": "http://www.brigade.petlifeline.in",
        "map_url": "https://maps.app.goo.gl/NXqQLJqDouJxYdN86",
        "image": "https://lh3.googleusercontent.com/gps-cs-s/AG0ilSw1zhtx8BfGOv5Mb0MmvBRrJ8B3KO0uIGxWE2njzB_p_r-jdfDrYo9E46Xew8sAjQhvR6EPeDdgWVJVwoqK_jIM92L1bG8xtcjtkHpljqPGobXSlupJGwh-0pNVwBy9TFnQTbE-=w408-h306-k-no"
    },
    {
        "name": "PfA Pet Care & Clinic",
        "phone": "+91 95132 68686",
        "address": "Uttarahalli Main Rd, Next to BGS Hospitals, Kengeri, Bengaluru 560060",
        "rating": 4.9,
        "reviews": 131,
        "open_close": "9:00 AM – 7:30 PM (Daily)",
        "website": "https://www.pfapetclinic.com",
        "map_url": "https://maps.app.goo.gl/Yt2WmTCbTEfFYriK7",
        "image": "https://lh3.googleusercontent.com/p/AF1QipNWI4wlujD-nEPaIc_Mqn3Byx9w5wtkaH9gooPo=w408-h725-k-no"
    },
    {
        "name": "Pride Pets Health & Wellness Center",
        "phone": "+91 9731144556",
        "address": "Dr. Vishnu Vardhana Road, Rajarajeshwari Nagar, Bengaluru 560098",
        "rating": 5.0,
        "reviews": 33,
        "open_close": "10:00 AM – 8:30 PM",
        "website": "http://www.japethospital.com",
        "map_url": "https://maps.app.goo.gl/isS4ybn6zF9K8HFV6",
        "image": "https://lh3.googleusercontent.com/p/AF1QipOmI66ujS5EiBmc_Z_Wjz_VskJNuznbaBfrz-3d=w408-h306-k-no"
    },
]

def show_hospitals_static():
    st.subheader("Nearby Veterinary Hospitals (Based on RNSIT Locality)")
    for h in HOSPITALS_DATA:
        st.markdown(f"### {h['name']}")
        if h.get("image"):
            st.image(h["image"], width=220)
        st.write(f"📍 Address: {h['address']}")
        st.write(f"📞 Phone: {h['phone']}")
        st.write(f"🕐 Hours: {h['open_close']}")
        st.write(f"⭐ Rating: {h['rating']} ({h['reviews']} reviews)")

        links = []
        if h.get("website"):
            links.append(f"[Website]({h['website']})")
        if h.get("map_url"):
            links.append(f"[Google Maps]({h['map_url']})")
        st.markdown(" • ".join(links))
        st.markdown("---")

# Usage
show_hospitals_static()
