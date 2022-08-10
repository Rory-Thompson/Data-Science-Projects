--****PLEASE ENTER YOUR DETAILS BELOW****
--mh-queries.sql

--Student ID: 31479839
--Student Name: Rory Thompson
--Tutorial No: 5

/* Comments for your marker:

    Q1
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer
SELECT
    ht_nbr,
    mh.endorsement.emp_nbr,
    emp_lname,
    emp_fname,
    to_char(end_last_annual_review, 'Dy DD Mon YYYY') review_date
FROM
         mh.endorsement
    INNER JOIN mh.employee ON mh.employee.emp_nbr = mh.endorsement.emp_nbr
WHERE
    end_last_annual_review > DATE '2020-03-31'
ORDER BY
    mh.endorsement.end_last_annual_review;


/*
    Q2
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer
--charter where special treatment is required

SELECT
    charter_nbr,
    mh.charter.client_nbr,
    client_lname,
    client_fname,
    charter_special_reqs
FROM
         mh.charter
    INNER JOIN mh.client ON mh.client.client_nbr = mh.charter.client_nbr
WHERE
    mh.charter.charter_special_reqs IS NOT NULL
ORDER BY
    mh.charter.charter_nbr;
/*
    Q3
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer
--need destination, charter cost per hour,
-- need to show the charter number, client last name, client first name listed as a single column
--use the in statement with the subquery that has all charter_nbrsfor all charter legs passing the condition
SELECT
    ch.charter_nbr, trim(cl.client_fname
    || ' '
    || cl.client_lname) AS fullname, ch.charter_cost_per_hour
FROM mh.charter ch 
inner join mh.client cl on cl.client_nbr = ch.client_nbr
    where 
        (ch.charter_cost_per_hour < 1000 or charter_special_reqs is NULL)
        and 
    ch.charter_nbr in ( select chl.charter_nbr 
            from mh.charter_leg chl 
                inner join mh.location loc on loc.location_nbr = chl.location_nbr_destination
            where loc.location_name = 'Mount Doom')
order by fullname desc
;

/*
    Q4
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer
select mh.helicopter.ht_nbr, mh.helicopter_type.ht_name, count(*) num_heli_owned
from mh.helicopter
inner join mh.helicopter_type on mh.helicopter_type.ht_nbr = mh.helicopter.ht_nbr
group by
mh.helicopter.ht_nbr, mh.helicopter_type.ht_name
having count(*) >= 2
order by
count(*) DESC;



/*
    Q5
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer
--need location name and number and the number of times a location has been used as an origin
select mh.charter_leg.location_nbr_origin, mh.location.location_name, count(*) num_as_origin
from mh.charter_leg
inner join mh.location on mh.location.location_nbr = mh.charter_leg.location_nbr_origin
group by
mh.charter_leg.location_nbr_origin, mh.location.location_name
having count(*) > 1
order by
count(*);
/*
    Q6
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer
--need the helicopter type and number
select mh.helicopter_type.ht_nbr, mh.helicopter_type.ht_name, 
    nvl(sum(heli_hrs_flown),0)  heli_type_hrs_flown
from mh.helicopter
right outer join mh.helicopter_type on mh.helicopter_type.ht_nbr = mh.helicopter.ht_nbr
group by
mh.helicopter_type.ht_nbr, mh.helicopter_type.ht_name
order by nvl(sum(heli_hrs_flown),0) 

;
/*
    Q7
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer

-- find all charters join charter leg with charter with employee

select ch.charter_nbr,
    (select ch2.cl_atd from mh.charter_leg ch2 where ch2.cl_leg_nbr =1 and ch2.charter_nbr = ch.charter_nbr) date_time_dep_leg_1
    from MH.charter ch 
        inner join mh.employee e on e.emp_nbr = ch.emp_nbr
    where ch.charter_nbr  not in (select chl.charter_nbr from mh.charter_leg chl where chl.cl_ata is null)
        and trim(e.emp_fname || ' ' || e.emp_lname) = 'Frodo Baggins'
    order by (select ch2.cl_atd from mh.charter_leg ch2 where ch2.cl_leg_nbr =1 and ch2.charter_nbr = ch.charter_nbr) desc
;

/*
    Q8
*/
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
-- ENSURE your query has a semicolon (;) at the end of this answer

--compare the average to the actual, need to use lpad to move it accross
--use cl_atd and cl_ata, actual time of departure and arrival
--find the average of the group by sum query
-- just minus the values, compare each total cost to the average cost of total costs

select leg.charter_nbr, cl.client_nbr, NVL(cl.client_lname, '-') client_last_name, NVL(cl.client_fname, '-') client_first_name,
    lpad(to_char(round(sum( 24 * (leg.cl_ata - leg.cl_atd) * ch.charter_cost_per_hour),2),'$999,999.99'),17,' ') totalchartercost
from mh.charter_leg leg
    inner join mh.charter ch on ch.charter_nbr = leg.charter_nbr
    inner join mh.client cl on cl.client_nbr = ch.client_nbr
group by cl.client_nbr, cl.client_lname, cl.client_fname, leg.charter_nbr
having sum( 24 * (leg.cl_ata - leg.cl_atd) * ch.charter_cost_per_hour) < 
    (select avg( sum( 24 * (leg.cl_ata - leg.cl_atd) * ch.charter_cost_per_hour))
        from mh.charter_leg leg
            inner join mh.charter ch on ch.charter_nbr = leg.charter_nbr
            inner join mh.client cl on cl.client_nbr = ch.client_nbr
            group by leg.charter_nbr
            )
order by totalchartercost desc

;

--q.9
-- PLEASE PLACE REQUIRED SQL STATEMENT FOR THIS PART HERE
--make two sub queries, one of the total charters completed on time, and the one only when it is same actual to estimated
--then compare these in a new select statement
select  mh.charter_leg.charter_nbr, trim(e.emp_fname || ' ' || e.emp_lname) pilotname, 
        trim(cl.client_fname || ' ' || cl.client_lname) clientname
    from mh.charter_leg
    inner join mh.charter c on c.charter_nbr = mh.charter_leg.charter_nbr
    inner join mh.client cl on cl.client_nbr = c.client_nbr
    inner join mh.employee e on e.emp_nbr = c.emp_nbr
    group by mh.charter_leg.charter_nbr, e.emp_fname, e.emp_lname, cl.client_fname,cl.client_lname
    having sum(case when charter_leg.cl_etd = charter_leg.cl_atd then 1 else 0 end) =  count(1)
    order by mh.charter_leg.charter_nbr
    ;


-- ENSURE your query has a semicolon (;) at the end of this answer
