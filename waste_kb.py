"""Curated waste knowledge base. Facts are shown even if the LLM output is cut,
and are injected into prompts so Llama only has to PHRASE, not RECALL, facts."""
KB = {
 "plastic": dict(name="Plastic (PET/HDPE bottles, containers)", bin="Dry waste - Blue bin",
  decomp="PET bottle: ~450 years; plastic bag: 10-20 years (thin film), carry-bag up to 500+",
  steps=["Empty and rinse the container","Remove cap and label if possible","Crush to save space","Put in dry/recyclable bin"],
  recycle="PET/HDPE are recycled into fibre, crates, pipes and new bottles. Look for resin codes 1 and 2.",
  impact="Plastic breaks into microplastics that enter water and food chains.",
  tip="Carry a reusable bottle and cloth bag."),
 "paper": dict(name="Paper & Cardboard", bin="Dry waste - Blue bin",
  decomp="Paper: 2-6 weeks; cardboard: ~2 months (when wet)",
  steps=["Keep it dry and clean","Flatten boxes","Remove tape and staples","Do not recycle greasy/food-soiled paper"],
  recycle="Pulped and made into new paper, egg trays and packaging; each fibre can be reused 5-7 times.",
  impact="Recycling 1 tonne of paper saves roughly 17 trees and large amounts of water and energy.",
  tip="Print double-sided; go digital."),
 "glass": dict(name="Glass bottles & jars", bin="Dry waste - Blue bin (or glass collection point)",
  decomp="Glass: ~1,000,000 years (effectively never)",
  steps=["Rinse and remove lids","Do not mix with ceramics or window glass","Wrap broken glass in paper and mark 'GLASS'"],
  recycle="Glass is 100% recyclable endlessly without quality loss; melted into new bottles.",
  impact="Recycled glass cuts furnace energy and CO2 emissions and saves raw sand.",
  tip="Reuse jars for storage."),
 "metal": dict(name="Metal (aluminium/tin cans)", bin="Dry waste - Blue bin",
  decomp="Aluminium can: 80-200 years; tin can: ~50 years",
  steps=["Empty and rinse","Crush cans","Keep separate from food waste"],
  recycle="Aluminium recycling uses ~95% less energy than making new metal.",
  impact="Mining new metal causes habitat loss and heavy emissions.",
  tip="Sell scrap metal to local kabadiwala / recycler."),
 "organic": dict(name="Organic / Food waste", bin="Wet waste - Green bin",
  decomp="Fruit/veg peels: 2-5 weeks; cooked food: 1-4 weeks",
  steps=["Drain liquids","Use a compostable liner or none","Do not mix with plastic","Home-compost or give to municipal wet-waste collection"],
  recycle="Composting gives nutrient-rich manure; biogas plants turn it into cooking gas.",
  impact="In landfills food waste produces methane, a potent greenhouse gas.",
  tip="Plan meals, store food well, compost at home."),
 "ewaste": dict(name="E-waste (phones, chargers, circuit boards)", bin="Special - authorised e-waste collection centre",
  decomp="Plastic casing/boards: hundreds of years; toxic metals never degrade",
  steps=["Never put in household bin","Wipe personal data","Remove batteries if possible","Hand over to authorised e-waste recycler or take-back programme"],
  recycle="Recovers gold, copper, silver and rare earths ('urban mining').",
  impact="Lead, mercury and cadmium leach into soil and groundwater if dumped.",
  tip="Repair, donate or trade-in before discarding."),
 "battery": dict(name="Batteries", bin="Hazardous - battery drop box / e-waste centre",
  decomp="100+ years; casings corrode and leak heavy metals",
  steps=["Tape terminals of lithium cells","Store in a dry container","Never burn or puncture","Drop at collection point"],
  recycle="Lithium, cobalt, nickel and lead are recovered for new batteries.",
  impact="Fire hazard in garbage trucks; leaked metals poison water.",
  tip="Use rechargeable batteries."),
 "textile": dict(name="Textiles / Clothing", bin="Donate or textile recycling",
  decomp="Cotton: 1-5 months; polyester/nylon: 20-200 years",
  steps=["Wash and donate wearable items","Reuse old cloth as rags","Give unusable fabric to textile recyclers"],
  recycle="Fibres are shredded into insulation, stuffing and new yarn.",
  impact="Fast fashion is a major source of water pollution and microfibres.",
  tip="Buy less, choose durable clothes."),
 "hazardous": dict(name="Hazardous (paint, chemicals, bulbs, medicines)", bin="Hazardous - authorised collection only",
  decomp="Varies: chemicals and heavy metals persist for decades or longer",
  steps=["Keep in original container","Never pour down drains","Hand over to hazardous-waste drive or pharmacy take-back"],
  recycle="Specialised facilities neutralise or recover materials safely.",
  impact="Contaminates soil and water; harms workers and wildlife.",
  tip="Buy only what you need."),
 "styrofoam": dict(name="Thermocol / Styrofoam, cigarette butts, chip packets", bin="Reject/Residual waste - Black bin",
  decomp="Styrofoam: 500+ years; cigarette butt: 10-12 years; multilayer packets: 100+ years",
  steps=["Keep clean and dry","Do not burn (toxic fumes)","Use only as last resort; refuse single-use items"],
  recycle="Rarely recycled locally; some EPR schemes collect multilayer plastic.",
  impact="Fragments easily and is ingested by animals.",
  tip="Refuse single-use thermocol plates and cups."),
}
# keyword fragments from ImageNet / manual labels -> category
KEYMAP = {
 "plastic":["water bottle","pop bottle","plastic bag","bottle","jug","container","tub","bucket","detergent"],
 "paper":["carton","envelope","paper","book","newspaper","menu","packet","crate","cardboard","box"],
 "glass":["glass bottle","glass jar","wine bottle","beer bottle","jar"],
 "metal":["can","tin","pot","pan","wok","bowl","spatula","ladle","cleaver"],
 "organic":["banana","orange","apple","lemon","strawberry","pineapple","pomegranate","fig","mushroom","broccoli","cucumber","corn","pizza","bread","bagel","burrito","vegetable","fruit"],
 "ewaste":["cellular","phone","mobile","smartphone","laptop","notebook","mouse","keyboard","monitor","screen","remote","charger","modem","hard disc","printer","circuit","pcb","speaker","camera","television","computer","tablet","cable","wire"],
 "battery":["battery","cell"],
 "textile":["jersey","sock","jean","shirt","cloth","sweatshirt","cardigan","pajama","towel","sandal"],
 "hazardous":["syringe","pill","lighter","spray","lotion","sunscreen","paint"],
 "styrofoam":["cup","plate","cigarette","tray","foam"],
}
import re, json, os
def _load():
    """Merge labels_map.json so ANY dataset's class names (TrashNet, TACO, Kaggle Garbage-12, your own) map to a category."""
    KEYMAP["plastic"] = [k for k in KEYMAP["plastic"] if k not in ("bottle", "container")]  # generic bottle material is ambiguous
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels_map.json")
    if os.path.exists(p):
        for c, ks in json.load(open(p)).items():
            if c in KB: KEYMAP[c] = list(dict.fromkeys(list(KEYMAP.get(c, [])) + ks))
_load()
def map_label(label):
    """Longest whole-word keyword match wins (so 'glass bottle' -> glass, 'plastic bottle cap' -> plastic)."""
    l = re.sub(r"[_\-]+", " ", label.lower()); best, bl = None, 0
    for cat, keys in KEYMAP.items():
        for k in keys:
            if len(k) > bl and re.search(r"\b" + re.escape(k) + r"s?\b", l): best, bl = cat, len(k)
    return best
