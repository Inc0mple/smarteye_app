CREATE TABLE IF NOT EXISTS EventLog (
    userId INTEGER NOT NULL,
    deviceId TINYINT(2) NOT NULL,
    timestamp TIMESTAMP DEFAULT (datetime('now', 'localtime')),
    eventId TINYINT(2) NOT NULL,
    PRIMARY KEY (userId, deviceId, timestamp)
);
INSERT INTO EventLog 
VALUES (1,0,datetime('now', 'localtime'),0);