| contract | clause | basis | guarantee | examples | misreadings (rejected by) | status |
|---|---|---|---|---|---|---|
| ADAMW | AW.C1 | S1.A1 | exact | E1:established | M1->E1/E2/E3/E4, M2->E1/E2/E3/E4 | established |
| ADAMW | AW.C2 | S1.A2 | exact | E1:established, E4:established | M2->E1/E2/E3/E4 | established |
| ADAMW | AW.C3 | S1.A3 | exact | E1:established, E2:established, E4:established | M3->E2/E3/E4 | established |
| ADAMW | AW.C4 | S1.A4 | rigorous_enclosure | E1:established, E3:established | M4->E3 | established |
| CE | CE.C1 | S1.D1 | rigorous_enclosure | E1:established, E2:established | - | established |
| CE | CE.C2 | S1.D1+S1.D2+S1.D5 | exact | E3:established | M4->E3 | established |
| CE | CE.C3 | S1.D2 | exact | E3:established | M1->E3, M2->E3 | established |
| CE | CE.C4 | S1.D6 + decision CE-A1 (reading R_A) | rigorous_enclosure | E4:established, E5:established | M3->E4/E5 | established |
| CE | CE.C5 | decision CE-A3 | allowed_set | E6:established | - | established |
| RANKMEAN | RM.C1 | T1.Q1 | exact | E1:established, E2:established | M1->E1, M2->E1/E2, M3->E1/E2 | established |
| RANKMEAN | RM.C2 | decision (no valid token) | allowed_set | E3:established | - | established |

ADAMW: clauses=4 examples=4 misreadings rejected 4/4; schema problems=none; reading checks=-
CE: clauses=5 examples=6 misreadings rejected 4/4; schema problems=none; reading checks={'E5[R_B]': 'established'}
RANKMEAN: clauses=2 examples=3 misreadings rejected 3/3; schema problems=none; reading checks=-
