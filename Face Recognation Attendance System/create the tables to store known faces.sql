 create the tables to store known faces and attendance logs.


create table ass3.detected_faces
(
    id            int auto_increment
        primary key,
    detected_name varchar(255) null,
    timestamp     timestamp    null
);
