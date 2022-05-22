## Column information

Column Info:   
- **Unnamed 0**:    
    Integer (Non-Null, ordinal) - Running play count per game, 0 indexed.
- **EVENTMSGACTIONTYPE**:   
    Integer (Non-Null, nominal) - Sub-ID of EVENTMSGTYPE qualifying the play further (e.g. Jump Shot or Layup for Field Goal Made)
- **EVENTMSGTYPE**:     
    Integer (Non-Null, nominal) - ID of play type (e.g. Rebound, Field Goal Made etc.). 14 different values.
- **EVENTNUM**:    
    Integer (Non-Null, somewhat ordinal) - Seems to be some running ID of the plays in a game. Has skips and not entirely in order (unlike column Unamed 0).
- **GAME_ID**:  
    String (Non-Null, ordinal) - Unique ID of a game. "XXXYYGGGGG", where XXX is the game type (in this dataset always 002 for regular season game). YY is the season year. GGGGGG is the game indentifier in the season, ordinal.
- **HOMEDESCRIPTION**:  
    String (text) - Loosely structured raw textual description of the play from the home team's perspective.
- **NEUTRALDESCRIPTION**:  
    Null - Empty Column
- **PCTIMESTRING**:  
    String (Non-Null, ordinal) - Game Clock, Time left in quarter.
- **PERIOD**:  
    Integer (Non-Null, ordinal) - Current period/quarter/overtime.
- **PERSON1TYPE**:  
    Integer/Float (nominal) - Property of player 1: e.g. 4.0: "Hometeam" (needs resolving)
- **PERSON2TYPE**:  
    Integer/Float (nominal) - Property of player 2 (needs resolving)
- **PERSON3TYPE**:  
    Integer/Float (nominal) - Property of player 3 (needs resolving)
- **PLAYER1_ID**:  
    Integer - ID of player 1 involved in the play. Can also be a team ID if no player was directly involved. 0 if neither team directly involved (e.g. Start of quarter)
- **PLAYER1_NAME**:  
    String - Name of player 1
- **PLAYER1_TEAM_ABBREVIATION  
    String - Abbreviation of player 1 team name
- **PLAYER1_TEAM_CITY**:  
    String - City of player 1 team
- **PLAYER1_TEAM_ID**:  
    Integer/Float - ID of player 1 team
- **PLAYER1_TEAM_NICKNAME**:  
    String - Short form of player 1 team name
- **PLAYER2_ID**:  
    Integer - ID of player 2 involved in the play. Can also be a team ID if no player was directly involved. 0 if neither team directly involved (e.g. Start of quarter)
- **PLAYER2_NAME**:  
    String - Name of player 2
- **PLAYER2_TEAM_ABBREVIATION**:  
    String - Abbreviation of player 2 team name
- **PLAYER2_TEAM_CITY**:  
    String - City of player 2 team
- **PLAYER2_TEAM_ID**:  
    Integer/Float - ID of player 2 team
- **PLAYER2_TEAM_NICKNAME**:  
    String - Short form of player 2 team name
- **PLAYER3_ID**:  
    Integer - ID of player 3 involved in the play. Can also be a team ID if no player was directly involved. 0 if neither team directly involved (e.g. Start of quarter)
- **PLAYER3_NAME**:  
    String - Name of player 3
- **PLAYER3_TEAM_ABBREVIATION**:   
    String - Abbreviation of player 3 team name
- **PLAYER3_TEAM_CITY**:   
    String - City of player 3 team
- **PLAYER3_TEAM_ID**:   
    Integer/Float - ID of player 3 team
- **PLAYER3_TEAM_NICKNAME**:   
    String - Short form of player 3 team name
- **SCORE**:   
    String - Game score, only non-null if score changes or at end of quarter.
- **SCOREMARGIN**:   
    String - Score marging, positive value if home team leads, negative value if visitor team leads, "TIE" if the score is tied.
- **VISITORDESCRIPTION**:    
    String (text) - Loosely structured raw textual description of the play from the visiting team's perspective.
- **WCTIMESTRING**:   
    String (ordinal) - real world time.
    