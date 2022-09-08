"""Postgres SQL queries to retrieve ODiN data in various formats with different joins."""
# Query to filter out OPID's based on various criteria
year = 2019
opid_filter = f"""
SELECT DISTINCT o.opid
                   FROM odin.odin{year} o
                   WHERE (verpl = 1 AND (o.aankpc = 0 OR o.vertpc = 0 OR o.aankpc IS NULL OR o.vertpc IS NULL))
                   OR (o.hhgestinkg = 11 OR o.hhgestinkg is NULL)
                   OR (o.hhauto = 10 OR o.hhauto is NULL)
                   OR (o.weekdag IN (1, 7))
                   OR (o.vertuur < 5 OR o.vertuur > 22)
                   OR (o.aankuur < 5 OR o.aankuur > 22)
                   OR (o.betwerk = 4 OR o.betwerk is NULL)
                   OR (o.ovstkaart = 3 OR o.ovstkaart is NULL)
                   OR (o.oprijbewijsau = 2 OR o.oprijbewijsau is NULL) 
                   OR (o.herkomst = 4 OR o.herkomst is NULL)
                   OR (o.opbezitvm = 65 OR o.opbezitvm is NULL)
                   OR (o.brandstofepa1 IN (4, 5) OR o.brandstofepa1 is NULL)
                   OR (o.fqnefiets is NULL)
                   OR (o.fqefiets is NULL)
                   OR (o.doel in (4, 6)) -- Professional
                   OR (o.aantvpl = 1)
                   OR (o.opleiding in (5, 6))
"""

# FIXME include work and study from home (redennw 4 and 5 respectively) once there is new (2022) data available

odin = f"""
SELECT o.opid
       , o.leeftijd
       , o.opleiding
       , o.wopc AS home_pc4
       , o.wogem
       , o.sted as urbanized
       , o.geslacht
       , o.betwerk
       , o.ovstkaart
       , CASE WHEN o.oprijbewijsau = 1 THEN 1 ELSE 0 end as rijbewijs
       , o.herkomst
       , CASE WHEN o.opbezitvm IN (1, 2, 3) THEN 1 ELSE 0 END AS has_car
       , CASE 
            WHEN o.opbezitvm NOT IN (1, 2, 3) THEN 0 -- No car
            WHEN o.brandstofepa1 = 0 THEN 1 -- Internal combustion
            WHEN o.brandstofepa1 = 1 THEN 2 -- Electric
            WHEN o.brandstofepa1 IN (2, 3) THEN 3 -- Hybrid
            ELSE 0 END 
        AS type_car
       , CASE WHEN o.fqnefiets IN (1, 2, 3) THEN 1 ELSE 0 END AS has_bike
       , CASE WHEN o.fqefiets IN (1, 2, 3) OR o.fqbrsnor IN (1, 2, 3) THEN 1 ELSE 0 END AS has_ebike
       , CASE WHEN o.autodpart = 1 THEN 1 ELSE 0 end as car_sharing
       , o.hhsam
       , o.hhpers
       , o.hhlft4
       , o.hhgestinkg
       , o.hhauto
       , o.verplid
       , o.vertpc
       , o.aankpc
       , o.vertuur
       , o.aankuur
       , o.hvm
       , o.hvmrol
       , o.doel
       , o.jaar
       , o.op
       , o.verpl
FROM odin.odin{year} o
WHERE (verpl = 1 OR weggeweest = 0)
-- exclude opids with unknown origin/destination, income, work status, pt card, or car ownership
AND   opid NOT IN ({opid_filter})
-- exclude opids with non-standard trips (verpl > 1)
AND   opid NOT IN (SELECT t1.opid
                   FROM (SELECT o.opid , MAX(o.verpl) AS maxverplcode FROM odin.odin{year} o GROUP BY o.opid) t1
                   WHERE t1.maxverplcode > 1)
AND 1=1
ORDER BY opid
         , verplid
"""

# Only get a sample of 1000 OPID's
odin_debug = odin.replace(
    "AND 1=1", "AND opid IN (SELECT DISTINCT opid from odin.odin_all LIMIT 1000)"
)

# FIXME why weggeweest = 0?
