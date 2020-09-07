import cv2
import time
import numpy as np

BRIGHT = 80
CARD_BRIGHT = 50
font = cv2.FONT_HERSHEY_SIMPLEX

class Train:
    def __init__(self):
        self.img = []
        self.name = ""

class Q_card:
    def __init__(self):
        self.contour = []
        self.szerokosc = 0
        self.wysokosc = 0
        self.corner_pts = []
        self.srodek = []
        self.trans = []
        self.numer_img = [] # odp rozmiar numeru
        self.figura_img = [] # odp rozmiar figury
        self.numer_best = ""
        self.figura_best = ""
        self.numer_diff = 0 # numer_img - numer_best
        self.figura_diff = 0 # figura_img - figura_best

def prep_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    img_w, img_h = np.shape(image)[:2]
    thresh_level = gray[int(img_h / 100)][int(img_w / 2)] + BRIGHT
    ret, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh


def znajdz(thresh_image):
    ret, cnts, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    if len(cnts) == 0:
        return [], []

    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    for i in x:
        cnts_sort.append(cnts[i])
        hier_sort.append(hierarchy[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        obw = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * obw, True)

        if ((size < 120000) and (size > 25000)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


def prep_card(contour, image):
    qCard = Q_card()
    qCard.contour = contour

    obw = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * obw, True)
    pts = np.float32(approx)
    qCard.corner_pts = pts
    x, y, qCard.szerokosc, qCard.wysokosc = cv2.boundingRect(contour)

    avg = np.sum(pts, axis=0) / len(pts)
    srodek_x = int(avg[0][0])
    srodek_y = int(avg[0][1])
    qCard.srodek = [srodek_x, srodek_y]
    qCard.trans = flat(image, pts, qCard.szerokosc, qCard.wysokosc)
    Qcorner = qCard.trans[0:84, 0:32]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)
    thresh_level = Qcorner_zoom[15, int((32 * 4) / 2)] - CARD_BRIGHT

    if (thresh_level <= 0):
        thresh_level = 1

    ret, q_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)
    Qnumer = q_thresh[20:185, 0:128]
    Qfigura = q_thresh[186:336, 0:128]

    ret, Qnumer_cnts, hier = cv2.findContours(Qnumer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qnumer_cnts = sorted(Qnumer_cnts, key=cv2.contourArea, reverse=True)

    if len(Qnumer_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qnumer_cnts[0])
        Qnumer_roi = Qnumer[y1:y1 + h1, x1:x1 + w1]
        Qnumer_sized = cv2.resize(Qnumer_roi, (70, 125), 0, 0)
        qCard.numer_img = Qnumer_sized

    ret, Qfigura_cnts, hier = cv2.findContours(Qfigura, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qfigura_cnts = sorted(Qfigura_cnts, key=cv2.contourArea, reverse=True)

    if len(Qfigura_cnts) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Qfigura_cnts[0])
        Qfigura_roi = Qfigura[y2:y2 + h2, x2:x2 + w2]
        Qfigura_sized = cv2.resize(Qfigura_roi, (70, 100), 0, 0)
        qCard.figura_img = Qfigura_sized

    return qCard


def porownaj(qCard, train_numery, train_figury):
    numer_best_diff = 1000000
    figura_best_diff = 1000000
    numer_best_name = ""
    figura_best_name = ""

    if (len(qCard.numer_img) != 0) and (len(qCard.figura_img) != 0):
        diff_img = np.empty(qCard.numer_img.shape, qCard.numer_img.dtype)
        for t_numer in train_numery:
            diff_img = cv2.absdiff(qCard.numer_img, t_numer.img)
            numer_diff = int(np.sum(diff_img) / 255)

            if numer_diff < numer_best_diff:
                numer_best_diff = numer_diff
                numer_best_t_name = t_numer.name

        for t_figura in train_figury:
            diff_img = cv2.absdiff(qCard.figura_img, t_figura.img)
            figura_diff = int(np.sum(diff_img) / 255)

            if figura_diff < figura_best_diff:
                figura_best_diff = figura_diff
                figura_best_t_name = t_figura.name

    if (numer_best_diff < 2000):
        numer_best_name = numer_best_t_name

    if (figura_best_diff < 700):
        figura_best_name = figura_best_t_name

    return numer_best_name, figura_best_name, numer_best_diff, figura_best_diff


def rozwiazanie(image, qCard):
    x = qCard.srodek[0]
    y = qCard.srodek[1]
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    numer_name = qCard.numer_best
    cv2.putText(image, numer_name, (x - 60, y - 10), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, numer_name, (x - 60, y - 10), font, 1, (50, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(image, numer_name, (x - 60, y - 10), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    figura_name = qCard.figura_best
    cv2.putText(image, figura_name, (x - 60, y + 25), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, figura_name, (x - 60, y + 25), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    return image


def flat(image, pts, sz, h):
    temp = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=2)
    diff = np.diff(pts, axis=-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if sz <= 0.8 * h:  # pionowo
        temp[0] = tl
        temp[1] = tr
        temp[2] = br
        temp[3] = bl

    if sz >= 1.2 * h:  # poziomo
        temp[0] = bl
        temp[1] = tl
        temp[2] = tr
        temp[3] = br

    if sz > 0.8 * h and sz < 1.2 * h:  # other
        if pts[1][0][1] <= pts[3][0][1]:
            temp[0] = pts[1][0]
            temp[1] = pts[0][0]
            temp[2] = pts[3][0]
            temp[3] = pts[2][0]

        if pts[1][0][1] > pts[3][0][1]:
            temp[0] = pts[0][0]
            temp[1] = pts[3][0]
            temp[2] = pts[2][0]
            temp[3] = pts[1][0]

    max_szerokosc = 200
    max_wysokosc = 300

    dst = np.array([[0, 0], [max_szerokosc - 1, 0], [max_szerokosc - 1, max_wysokosc - 1], [0, max_wysokosc - 1]],
                   np.float32)
    M = cv2.getPerspectiveTransform(temp, dst)
    trans = cv2.warpPerspective(image, M, (max_szerokosc, max_wysokosc))
    trans = cv2.cvtColor(trans, cv2.COLOR_BGR2GRAY)

    return trans


videostream = cv2.VideoCapture(0);
time.sleep(1)

cam_quit = 0
readcards = []

train_numery = []
i = 0

for k in ['AS', 'Dwojka', 'Trojka', 'Czworka', 'Piatka', 'Szostka', 'Siodemka', 'Osemka', 'Dziewiatka', 'Dziesiatka', 'Walet', 'Dama', 'Krol']:
    train_numery.append(Train())
    train_numery[i].name = k
    train_numery[i].img = cv2.imread(k + '.jpg', cv2.IMREAD_GRAYSCALE)
    i = i + 1

train_figury = []
j = 0

for f in ['Pik', 'Karo', 'Trefl', 'Kier']:
    train_figury.append(Train())
    train_figury[j].name = f
    train_figury[j].img = cv2.imread(f + '.jpg', cv2.IMREAD_GRAYSCALE)
    j = j + 1

while cam_quit == 0:
    ret, image = videostream.read()
    pre_proc = prep_image(image)
    cnts_sort, is_card = znajdz(pre_proc)

    if len(cnts_sort) != 0:
        cards = []
        k = 0

        for i in range(len(cnts_sort)):
            if (is_card[i] == 1):
                cards.append(prep_card(cnts_sort[i],image))
                readcards.append(prep_card(cnts_sort[i], image))
                cards[k].numer_best, cards[k].figura_best, cards[k].numer_diff, cards[k].figura_diff = porownaj(cards[k],train_numery,train_figury)
                readcards[k].srodek = cards[k].srodek

                if (cards[k].numer_best != ""):
                    readcards[k].numer_best = cards[k].numer_best
                    readcards[k].numer_diff = cards[k].numer_diff

                if (cards[k].figura_best != ""):
                    readcards[k].figura_best = cards[k].figura_best
                    readcards[k].figura_diff = cards[k].figura_diff

                image = rozwiazanie(image, readcards[k])
                k = k + 1


        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)

    cv2.imshow("EX", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()
videostream.release()

