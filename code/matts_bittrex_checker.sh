#!/bin/bash
# install redis-server, redis-tools, mailutils
# set emails to notify. use text message email address for text message. space in between
emails=(matt.mock@gmail.com)
# run script 1x. then set welcome=0 below to get notif for new coins
welcome=0
curl https://bittrex.com/api/v1.1/public/getmarkets | sed 's#,{#\n#g' | while read line; do
rkey="bittrexnewcoin""$(echo -n "$line" | md5sum | sed 's# ##g' | sed 's#\n##g')"
if [ $(redis-cli sismember bittrex_notif_newcoins "$rkey") -eq 0 ]
then
        if [ $welcome -eq 0 ]
        then
                tempf=mktemp mail.XXXXXX
                echo "$line" > $tempf
                echo "sending notifications $line"
                for i in "${emails[@]}"
                do
                        mail -s "New Bittrex Coin!" "$i" < $tempf
                        echo "email sent to $i"
                done
                rm $tempf
        fi
        redis-cli sadd bittrex_notif_newcoins "$rkey"
fi
done
