<?php

include("./inclu/heed.php");
include("./include/function.php");

if(isset($_REQUEST['login'])){
    include("./inclu/db_conn.php");
    $firstname=$_REQUEST['firstname'];
    $lastname=$_REQUEST['lastname'];  
    $unitid=$_REQUEST['unitid'];
    $phonenumber=$_REQUEST['phonenumber'];
    $dateofbirth=$_REQUEST['dateofbirth'];
    $rank=$_REQUEST['rank'];
    $email=$_REQUEST['email'];
    $gender=$_REQUEST['gender'];
    $sql="INSERT INTO soldier_table(firstname,lastname,unitid,phonenumber,dateofbirth,rank,email,gender)
    VALUES('$firstname,','$lastname','$unitid','$phonenumber','$dateofbirth','$rank','$email','$gender')";
    if(mysqli_query($conn,$sql)){
        my_alert();
    }else{
        echo "Error :" .$sql."<br>".mysqli_error($conn);
    }
    mysqli_close($conn);
}
?>
    <div class="icon">
        <h2 class="logo">Soldiers information</h2>
    </div> 
    <form   method="post">
    <!-- <fieldset> -->
    <div class="container">
    Enter your firstname : <input type="text" name="firstname" placeholder="Enter your firstname" class="box-1"><br/>
    Enter your lastname : <input type="text" name="lastname" placeholder="Enter your lastname" class="box-2"><br/>
    Enter your unit ID : <input type="number" name="unitid" placeholder="Enter your unit ID" class="box-3"><br/>
    Enter your contact number :  <input type="number" name="phonenumber" placeholder="Enter your contact number" class="box-4"><br/>
    Enter your date of birth : <input type="date" size="30" name="dateofbirth" class="box-5"><br/>
    Enter your rank : <input type="number" name="rank" placeholder="Enter your rank" class="box-6"><br/>
    Enter your emial : <input type ="email" name = "email" placeholder = "Enter your email" class="box-7"><br/>
    Enter your gender : <input type="text" name="gender" placeholder="Enter Male/Female" class="box-7"></br>
    <button class="btnn"  type="submit" name="login" >Submit</button>
    <button class="btnn"><a href="combinedpage.php">Back</a></button>
    </div>
    <!-- </fieldset> -->
    </form>
<?php

include("./inclu/feet.php");

?>
